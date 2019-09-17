################################################################################
#
# Functional objective for FWI
#
################################################################################


export objFWI!


function objFWI!(F,
				 G::Union{Array{R, 1}, Nothing},
				 m::Array{R, 2},
				 h::NTuple{2, R},
				 freqs::Array{R, 1}, x_rcv::Array{R, 1}, z_rcv::R,
				 x_src::Array{R, 1}, z_src::R,
				 src_enc::Array{C, 2},
				 dat::Array{Array{C, 2}, 1};
				 nabc::Union{Nothing, NTuple{4, Int64}} = nothing,
				 gradprec_fun = g->g,
				 bounds::Union{NTuple{2, R}, NTuple{2, Array{R, 2}}, Nothing} = nothing)

    # Initializing objective value
    J = R(0)

    # Zero-ing gradient
    if G != nothing
        G .= R(0)
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

		# Data normalization factor
		ηd = sqrt.(sum(abs.(dat_enc).^R(2), dims = 1))

         # Set absorbing boundary size (frequency-dependent)
        if nabc == nothing
            nabc = computeNabc(h, m, freqs[idxf])
        end

        # Setting source/receiver restriction/injection operators & source weigths
        Pr = restrOpMat(n, h, x_rcv, z_rcv, nabc)
        Ps = restrOpMat(n, h, x_src, z_src, nabc)
        Fsrc = Matrix(adjoint(Ps)*src_enc)

        # Solve for wavefield (forward)
        H = lu(HelmholtzLinOp(h, m, freqs[idxf], nabc))
        U = C.(H\Fsrc)

        # Receiver restriction
        res = dat_enc-Pr*U

		# Objective value update
        if F != nothing
            J += (sum(sum(abs.(res).^R(2), dims = 1)./ηd.^R(2), dims = 2)/(Nf*Ns))[1]
        end

		# Gradient update
        if G != nothing
            V = C.(adjoint(H)\(adjoint(Pr)*res)) # Solve for wavefield (adjoint)
            g = -R(2).*(R(2).*π.*freqs[idxf]).^R(2).*restrict(reshape(sum(real.(conj.(U).*V)./ηd.^R(2), dims = 2), sz_abc(n, nabc)), nabc)./(Nf*Ns)
            G .+= gradprec_fun(g)
        end

    end

    if F != nothing
        return J
    end

end
