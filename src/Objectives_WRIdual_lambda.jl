################################################################################
#
# Functional objectives for WRIdual (λ version)
#
################################################################################


export objWRIdual_λ!, objWRIdual_λ_yopt!, objWRIdual_λ_yres!, objWRIdual_λ_yfixed!, objWRIdual_λ_mfixed!


function objWRIdual_λ!(F,
					   Gm::Union{Array{R, 1}, Nothing}, Gy::Union{Array{Array{C, 2}, 1}, Nothing},
					   m::Array{R, 2}, y::Union{Array{Array{C, 2}, 1}, Nothing},
					   h::NTuple{2, R},
					   freqs::Array{R, 1},
					   x_rcv::Array{R, 1}, z_rcv::R,
					   x_src::Array{R, 1}, z_src::R,
					   src_enc::Array{C, 2},
					   dat::Array{Array{C, 2}, 1},
					   λ::R;
					   nabc::Union{Nothing, NTuple{4, Int64}} = nothing,
					   α::Union{Nothing, R, Array{C, 2}} = nothing,
					   weight_array::Array{R, 2},
					   gradmprec_fun = g->g,
	  				   bounds::Union{NTuple{2, R}, NTuple{2, Array{R, 2}}, Nothing} = nothing,
					   mode_y::String = "provided")

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
		ηd = norm(dat_enc)
		ηf = norm(Fsrc)
		c = ηf./(λ.*ηd)

        # Helmholtz operator
        H = HelmholtzLinOp(h, m, freqs[idxf], nabc)
		if mode_y != "optimal"
			H = lu(H)
		end

		# Computing y, if not provided (y = data residual)
		if mode_y == "provided"
			y_ = y[idxf]
		elseif mode_y == "residual"
			Uinc = H\Fsrc
			y_ = dat_enc.-Pr*Uinc
		elseif mode_y == "optimal"
			W = spdiagm(0 => vec(weight_array))
			U = C.(ComplexF64.([c.*Pr; W*H])\[c.*dat_enc; W*Fsrc])
			y_ = dat_enc.-Pr*U
		end

		# Solve for wavefield (adjoint)
		if mode_y != "optimal"
			V = C.(adjoint(H)\(adjoint(Pr)*y_))
		else
			V = R(1)./c.^R(2).*vec(weight_array).^R(2).*((H*U).-Fsrc)
		end

		# ||F^T*y||_W^2 = <W*F^T*y, F^T*y>
		norm_FTy2 = sum(abs.(V./weight_array).^R(2), dims = 1)

		# <y, r>
		y_dot_r = sum(y_.*conj.(dat_enc), dims = 1)-sum(V.*conj.(Fsrc), dims = 1)

		# ||y||^2
		norm_y2 = sum(abs.(y_).^R(2), dims = 1)

		# Computing optimized α
		if α == nothing
			α_ = conj.(y_dot_r)./(c.^R(2).*norm_FTy2.+norm_y2)
		end

		# Objective value update
        if F != nothing
            L += (sum(-R(0.5).*abs.(α_).^R(2).*c.^R(2).*norm_FTy2.+real.(α_.*y_dot_r).-R(0.5).*abs.(α_).^R(2).*norm_y2, dims = 2)./Nf)[1]
			# L += (R(0.5).*sum(abs.(y_dot_r).^R(2)./(c.^R(2).*norm_FTy2.+norm_y2), dims = 2)./Nf)[1]
        end

		# Solve for wavefield (extended source)
        if (Gm != nothing || Gy != nothing) && mode_y != "optimal"
			Q = Fsrc.+c.^R(2).*α_.*V./weight_array.^R(2)
			U = C.(H\Q)
		end

		# Gradient update (y)
        if Gy != nothing || (Gm != nothing && mode_y == "residual")
            Gy_ = (conj.(α_).*(dat_enc.-Pr*U).-abs.(α_).^R(2).*y_)./Nf
        end
		if Gy != nothing
			Gy[idxf] = Gy_
		end

		# Gradient update (m)
		if Gm != nothing
            Gm_ = -(R(2)*π*freqs[idxf]).^R(2).*restrict(reshape(sum(real.(α_.*conj.(U).*V), dims = 2), sz_abc(n, nabc)), nabc)./Nf
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

function objWRIdual_λ_yopt!(F, Gm, m, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, λ; nabc = nothing, weight_array, gradmprec_fun = g->g, bounds = nothing)

	return objWRIdual_λ!(F, Gm, nothing, m, nothing, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, λ; nabc = nabc, α = nothing, weight_array = weight_array, gradmprec_fun = gradmprec_fun, bounds = bounds, mode_y = "optimal")

end

function objWRIdual_λ_yres!(F, Gm, m, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, λ; nabc = nothing, weight_array, gradmprec_fun = g->g, bounds = nothing)

	return objWRIdual_λ!(F, Gm, nothing, m, nothing, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, λ; nabc = nabc, α = nothing, weight_array = weight_array, gradmprec_fun = gradmprec_fun, bounds = bounds, mode_y = "residual")

end

function objWRIdual_λ_yfixed!(F, Gm, m, y, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, λ; nabc = nothing, α = nothing, weight_array, gradmprec_fun = g->g, bounds = nothing)

	return objWRIdual_λ!(F, Gm, nothing, m, y, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc, dat, λ; nabc = nabc, α = α, weight_array = weight_array, gradmprec_fun = gradmprec_fun, bounds = bounds, mode_y = "provided")

end

function objWRIdual_λ_mfixed!(F, Gy::Union{Nothing, Array{C, 2}}, m, y::Array{C, 2}, h, freq::R, x_rcv, z_rcv, x_src, z_src, src_enc, dat::Array{C, 2}, λ; nabc = nothing, α = nothing, weight_array, gradmprec_fun = g->g, bounds = nothing)

	if Gy == nothing
		return objWRIdual_λ!(F, nothing, nothing, m, [y], h, [freq], x_rcv, z_rcv, x_src, z_src, src_enc, [dat], λ; nabc = nabc, α = α, weight_array = weight_array, gradmprec_fun = gradmprec_fun, bounds = bounds, mode_y = "provided")
	else
		Gy_ = Array{Array{C, 2}, 1}(undef, 1)
		L = objWRIdual_λ!(F, nothing, Gy_, m, [y], h, [freq], x_rcv, z_rcv, x_src, z_src, src_enc, [dat], λ; nabc = nabc, α = α, weight_array = weight_array, gradmprec_fun = gradmprec_fun, bounds = bounds, mode_y = "provided")
		Gy .= -Gy_[1]
		return -L
	end

end
