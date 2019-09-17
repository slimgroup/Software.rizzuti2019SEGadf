################################################################################
#
# Functional objectives for WRIdual (ε version)
#
################################################################################


export objWRIdual_ε!, objWRIdual_ε_yres!, objWRIdual_ε_yfixed!, objWRIdual_ε_mfixed!


function objWRIdual_ε!(F,
					   Gm::Union{Array{R, 1}, Nothing}, Gy::Union{Array{Array{C, 2}, 1}, Nothing},
					   m::Array{R, 2}, y::Union{Array{Array{C, 2}, 1}, Nothing},
					   h::NTuple{2, R},
					   freqs::Array{R, 1},
					   x_rcv::Array{R, 1}, z_rcv::R,
					   x_src::Array{R, 1}, z_src::R,
					   src_enc::Array{C, 2},
					   dat::Array{Array{C, 2}, 1},
					   ε::Union{R, Array{Array{R, 2}, 1}};
					   nabc::Union{Nothing, NTuple{4, Int64}} = nothing,
					   α::Union{Nothing, R, Array{C, 2}} = nothing,
					   weight_array::Array{R, 2},
					   gradmprec_fun = g->g,
	  				   bounds::Union{NTuple{2, R}, NTuple{2, Array{R, 2}}, Nothing} = nothing,
					   mode_y::String = "provided",
					   mode_ε::String = "provided")

    # Initializing objective value
    L = R(0)

    # Zero-ing gradient
    if Gm != nothing
        Gm .= R(0)
    end

	# Projecting m to box constrained set
	m = proj_bounds(m, bounds)

    # Updating obj value and gradient (loop over frequencies)
    n = size(m)
    Ns = size(src_enc, 2)
    Nf = length(freqs)
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

		# Normalization factors
		ηd = sqrt.(sum(abs.(dat_enc).^R(2), dims = 1))
		ηf = sqrt.(sum(abs.(Fsrc).^R(2), dims = 1))

        # Helmholtz operator
        H = lu(HelmholtzLinOp(h, m, freqs[idxf], nabc)) # LU factorization

		# Computing y, if not provided (y = data residual)
		if mode_y == "provided"
			y_ = y[idxf]
		elseif mode_y == "residual"
			Uinc = C.(H\Fsrc)
			y_ = dat_enc.-Pr*Uinc
		end

		# Solve for adjoint wavefield
		V = C.(adjoint(H)\(adjoint(Pr)*y_))

		# ||F^T*y||_W^2 = <W*F^T*y, F^T*y>
		norm_FTy2 = sum(abs.(V./weight_array).^R(2), dims = 1)

		# <y, r>
		y_dot_r = sum(y_.*conj.(dat_enc), dims = 1)-sum(V.*conj.(Fsrc), dims = 1)

		# ||y||
		norm_y = sqrt.(sum(abs.(y_).^R(2), dims = 1))

		# Computing ε
		if mode_ε == "adaptive"
			if isa(ε, R)
				ε_ = ε*norm_y./ηd
			elseif isa(ε, Array{Array{R, 2}, 1})
				ε_ = ε[idxf].*norm_y./ηd
			end
		elseif mode_ε == "provided"
			if isa(ε, R)
				ε_ = ε
			elseif isa(ε, Array{Array{R, 2}, 1})
				ε_ = ε[idxf]
			end
		end

		# Computing optimized α
		if α == nothing
			α_ = (abs.(y_dot_r) .>= ε_.*ηd.*norm_y).*(abs.(y_dot_r).-ε_.*ηd.*norm_y)./(ηf.^R(2).*norm_FTy2).*conj.(y_dot_r)./abs.(y_dot_r)
		else
			α_ = α
		end

		# Objective value update
        if F != nothing
			L += (sum(-R(0.5).*abs.(α_).^R(2).*ηf.^R(2).*norm_FTy2.+real.(α_.*y_dot_r).-ε_.*ηd.*abs.(α_).*norm_y, dims = 2)/(Nf*Ns))[1]
        end

		# Solve for wavefield (extended source)
        if Gm != nothing || Gy != nothing
			Q = Fsrc.+ηf.^R(2).*α_.*V./weight_array.^R(2)
			U = C.(H\Q)
		end

		# Gradient update (y)
        if Gy != nothing || (Gm != nothing && mode_y == "residual")
            Gy_ = (conj.(α_).*(dat_enc-Pr*U)-ε_.*abs.(α_).*ηd.*y_./norm_y)./(Nf*Ns)
        end
		if Gy != nothing
			Gy[idxf] = Gy_
		end

		# Gradient update (m)
		if Gm != nothing
            Gm_ = -(R(2)*π*freqs[idxf]).^R(2).*restrict(reshape(sum(real.(α_.*conj.(U).*V), dims = 2), sz_abc(n, nabc)), nabc)./(Nf*Ns)
			if mode_y == "residual"
				VGy = C.(adjoint(H)\(adjoint(Pr)*Gy_))
				Gm_ .+= -(R(2)*π*freqs[idxf]).^R(2).*restrict(reshape(sum(real.(conj.(Uinc).*VGy), dims = 2), sz_abc(n, nabc)), nabc)
			end
            Gm .+= gradmprec_fun(Gm_)
        end

    end

	if F != nothing
        return L
    end

end

function objWRIdual_ε_yres!(F, G, m, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, ε; nabc = nothing, α = nothing, weight_array, gradmprec_fun = g->g, bounds = nothing, mode_ε = "provided")

	return objWRIdual_ε!(F, G, nothing, m, nothing, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, ε; nabc = nabc, α = α, weight_array = weight_array, gradmprec_fun = gradmprec_fun, bounds = bounds, mode_y = "residual", mode_ε = mode_ε)

end

function objWRIdual_ε_yfixed!(F, Gm, m, y, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, ε; nabc = nothing, α = nothing, weight_array = weight_array, gradmprec_fun = g->g, bounds = nothing, mode_ε = "provided")

	return objWRIdual_ε!(F, Gm, nothing, m, y, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, ε; nabc = nabc, α = α, weight_array = weight_array, gradmprec_fun = gradmprec_fun, bounds = bounds, mode_y = "provided", mode_ε = mode_ε)

end

function objWRIdual_ε_mfixed!(F, Gy::Union{Array{C, 2}, Nothing}, m, y::Array{C, 2}, h, freq::R, x_rcv, z_rcv, x_src, z_src, src_enc, dat::Array{C, 2}, ε; nabc = nothing, α = nothing, weight_array = weight_array, bounds = nothing, mode_ε = "provided")

	if Gy == nothing
		return objWRIdual_ε!(F, nothing, nothing, m, [y], h, [freq], x_rcv, z_rcv, x_src, z_src, src_enc, [dat], ε; nabc = nabc, α = α, weight_array = weight_array, bounds = bounds, mode_y = "provided", mode_ε = mode_ε)
	else
		Gy_ = Array{Array{C, 2}, 1}(undef, 1)
		L = objWRIdual_ε!(F, nothing, Gy_, m, [y], h, [freq], x_rcv, z_rcv, x_src, z_src, src_enc, [dat], ε; nabc = nabc, α = α, weight_array = weight_array, bounds = bounds, mode_y = "provided", mode_ε = mode_ε)
		Gy .= -Gy_[1]
		return L
	end

end
