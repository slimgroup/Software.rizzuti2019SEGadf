################################################################################
#
# General utilities
#
################################################################################


export extend, restrict, sz_abc, gaussian_filt, contr2abs, abs2contr, gradprec_contr2abs, gendata, compute_y, proj_bounds


function sz_abc(n::NTuple{2, Int64}, nabc::NTuple{4, Int64})

    return n.+(nabc[1]+nabc[2], nabc[3]+nabc[4])

end


function extend(m::Array{T, 2}, nabc::NTuple{4, Int64}) where T <: RuC

    m_ext = Array{T}(undef, sz_abc(size(m), nabc))
    m_ext[nabc[1]+1:end-nabc[2], nabc[3]+1:end-nabc[4]] .= m
    m_ext[1:nabc[1], :] .= reshape(m_ext[nabc[1]+1, :], 1, :) # left
    m_ext[end-nabc[2]+1:end, :] .= reshape(m_ext[end-nabc[2], :], 1, :) # right
    m_ext[:, 1:nabc[3]] .= reshape(m_ext[:, nabc[3]+1], :, 1) # top
    m_ext[:, end-nabc[4]+1:end] .= reshape(m_ext[:, end-nabc[4]], :, 1) # bottom

    return m_ext

end

function extend(m::Array{T, 3}, nabc::NTuple{4, Int64}) where T <: RuC

    n = (size(m, 1), size(m, 2))
    m_ext = Array{T}(undef, (sz_abc(size(m), nabc)..., size(m, 3)))
    m_ext[nabc[1]+1:end-nabc[2], nabc[3]+1:end-nabc[4], :] .= m
    m_ext[1:nabc[1], :, :] .= reshape(m_ext[nabc[1]+1, :, :], 1, size(m_ext, 2), :) # left
    m_ext[end-nabc[2]+1:end, :, :] .= reshape(m_ext[end-nabc[2], :, :], 1, size(m_ext, 2), :) # right
    m_ext[:, 1:nabc[3], :] .= reshape(m_ext[:, nabc[3]+1, :], :, 1, size(m_ext, 3)) # top
    m_ext[:, end-nabc[4]+1:end, :] .= reshape(m_ext[:, end-nabc[4]], :, 1, size(m_ext, 3)) # bottom

    return m_ext

end


## Restriction for 2D images

function restrict(m::Array{T, 2}, nabc::NTuple{4, Int64}) where T <: RuC

    return m[nabc[1]+1:end-nabc[2], nabc[3]+1:end-nabc[4]]

end

function restrict(m::Array{T, 3}, nabc::NTuple{4, Int64}) where T <: RuC

    return m[nabc[1]+1:end-nabc[2], nabc[3]+1:end-nabc[4], :]

end


## Utilities for model input preprocessing and gradient postprocessing: contrast to absolute properties / effective to total domain

# Contrast preconditioning
function contr2abs(xvec::Array{R, 1}, mask::BitArray{2}, mb::Array{R, 2})
# Preprocessing steps: - effective to global domain
#                      - contrast to absolute properties

    x = zeros(R, size(mb))
    x[mask] .= xvec
    return mb.*(R(1).+x)

end
function abs2contr(m::Array{R, 2}, mask::BitArray{2}, mb::Array{R, 2})

	return ((m.-mb)./mb)[mask]

end

function gradprec_contr2abs(grad::Array{R, 2}, mask::BitArray{2}, mb::Array{R, 2})
# Postprocessing steps: - preconditioning by diagonal background model

    return mb[mask].*grad[mask]

end


## Smoothing utilities

function gaussian_filt(m_raw::Array{R, 2}, σ::R)

    return imfilter(m_raw, Kernel.gaussian(σ))

end


## Generate data

function gendata(m::Array{R, 2}, h::NTuple{2, R}, freqs::Array{R, 1}, x_rcv::Array{R, 1}, z_rcv::R, x_src::Array{R, 1}, z_src::R, src_enc::Array{C, 2}; nabc::Union{Nothing, NTuple{4, Int64}} = nothing, dat::Union{Nothing, Array{Array{C, 2}, 1}} = nothing, verbose::Bool = false)

	# Initializing data output
	n = size(m)
	Nr = length(x_rcv)
	Ns = size(src_enc, 2)
	Nf = length(freqs)
	rec = Array{Array{C, 2}, 1}(undef, Nf)

    # Modeling (loop over frequencies)
    for idxf = 1:Nf

		if verbose
			println("Generating data for frequency: ", freqs[idxf], " Hz")
		end

		# Set absorbing boundary size (frequency-dependent)
        if nabc == nothing
            nabc = computeNabc(h, m, freqs[idxf])
        end

        # Setting source/receiver restriction/injection operators & source weigths
        Pr = restrOpMat(n, h, x_rcv, z_rcv, nabc)
        Ps = restrOpMat(n, h, x_src, z_src, nabc)
        Fsrc = Matrix(adjoint(Ps)*src_enc)

        # Solve for wavefield (forward)
        H = HelmholtzLinOp(h, m, freqs[idxf], nabc)
        U = C.(H\Fsrc)

        # Receiver restriction
		if dat == nothing
			rec[idxf] = Pr*U # Data recorded
		else
			rec[idxf] = dat[idxf]*src_enc.-Pr*U # Data residual
		end

    end

	return rec

end

function compute_y(m::Array{R, 2}, h::NTuple{2, R}, freqs::Array{R, 1}, x_rcv::Array{R, 1}, z_rcv::R, x_src::Array{R, 1}, z_src::R, src_enc::Array{C, 2}, dat::Array{Array{C, 2}, 1}, λ::R; nabc::Union{Nothing, NTuple{4, Int64}} = nothing, weight_array::Array{R, 2}, mode_y::String = "optimal", c::Union{Nothing, Array{R, 1}} = nothing, β::Union{R, Nothing} = nothing)

	# Initializing data output
	n = size(m)
	Nr = length(x_rcv)
	Ns = size(src_enc, 2)
	Nf = length(freqs)
	y = Array{Array{C, 2}, 1}(undef, Nf)

	# Loop over frequencies
	for idxf = 1:Nf

		# Encoding data
		dat_enc = dat[idxf]*src_enc

         # Set absorbing boundary size (frequency-dependent)
        if nabc == nothing
            nabc = computeNabc(h, m, freqs[idxf])
        end

        # Setting source/receiver restriction/injection operators & source weigths
        Pr = restrOpMat(n, h, x_rcv, z_rcv, nabc)
        Ps = restrOpMat(n, h, x_src, z_src, nabc)
        Fsrc = Matrix(adjoint(Ps)*src_enc)

        # Helmholtz operator
        H = HelmholtzLinOp(h, m, freqs[idxf], nabc)
		if mode_y != "optimal"
			H = lu(H)
		end

		# Computing y, if not provided (y = data residual)
		if c == nothing && mode_y != "residual"
			ηd = norm(dat_enc)
			ηf = norm(Fsrc)
			c = ηf./(λ.*ηd)
		end
		if mode_y == "optimal"
			W = spdiagm(0 => vec(weight_array))
			U = C.(ComplexF64.([c.*Pr; W*H])\[c.*dat_enc; W*Fsrc])
			y[idxf] = dat_enc.-Pr*U
		elseif mode_y == "residual"
			U = C.(H\Fsrc)
			y[idxf] = dat_enc.-Pr*U
		elseif mode_y == "residual2"
			Winv2 = spdiagm(0 => vec(R(1)./weight_array.^R(2)))
			Uinc = C.(H\Fsrc)
			r = dat_enc.-Pr*Uinc
			Fr = C.(adjoint(H)\(adjoint(Pr)*r))
			FWFr = Pr*(C.(H\(Winv2*Fr)))
			y[idxf] = r.-β.*c.^R(2).*FWFr
		elseif mode_y == "adjoint"
			Winv2 = spdiagm(0 => vec(R(1)./weight_array.^R(2)))
			Uinc = C.(H\Fsrc)
			r = dat_enc.-Pr*Uinc
			Fr = C.(adjoint(H)\(adjoint(Pr)*r))
			FWFr = Pr*(C.(H\(Winv2*Fr)))
			y[idxf] = r.+c.^R(2).*FWFr
		end

	end

	return y

end


## Projection on box constraints

function proj_bounds(m::Array{R, 2}, bounds::NTuple{2, Array{R, 2}})
	m_ = copy(m)
	m_[m .< bounds[1]] .= bounds[1][m .< bounds[1]]
	m_[m .> bounds[2]] .= bounds[2][m .> bounds[2]]
	return m_
end

function proj_bounds(m::Array{R, 2}, bounds::NTuple{2, R})
	m_ = copy(m)
	m_[m .< bounds[1]] .= bounds[1]
	m_[m .> bounds[2]] .= bounds[2]
	return m_
end

function proj_bounds(m::Array{R, 2}, bounds::Nothing)
	return m
end
