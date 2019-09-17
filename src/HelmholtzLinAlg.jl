################################################################################
#
# Helmholtz-related linear algebra routines
#
################################################################################


export HelmholtzLinOp, computeNabc

function HelmholtzLinOp(h::NTuple{2, R}, m::Array{R, 2}, freq::R, nabc::NTuple{4, Int64})

    # Extension for ABC
    m_ext = extend(m, nabc)
    n_ext = size(m_ext)

    # Complex frequency for ABC
    abc_fact = 1f0
    ω = 2f0*pi*freq*(1f0.-abc_fact*1im*taper(size(m), nabc))

    # Laplacian
    Dxx = spdiagm(-1 => ones(R, n_ext[1]-1), 0 => -2f0*ones(R, n_ext[1]), 1 => ones(R, n_ext[1]-1))/h[1]^2
    Dxx[1, 1:2] = [1 -1]/h[1]^2 # Sommerfeld radiation condition
    Dxx[end, end-1:end] = [-1, 1]/h[1]^2 # Sommerfeld radiation condition
    Dzz = spdiagm(-1 => ones(R, n_ext[2]-1), 0 => -2f0*ones(R, n_ext[2]), 1 => ones(R, n_ext[2]-1))/h[2]^2
    Dzz[1, 1:2] = [1 -1]/h[2]^2 # Sommerfeld radiation condition
    Dzz[end, end-1:end] = [-1, 1]/h[2]^2 # Sommerfeld radiation condition
    Δ = kron(sparse(I, n_ext[2], n_ext[2]), Dxx)+kron(Dzz, sparse(I, n_ext[1], n_ext[1]))

    # Mass matrix
    w = ones(R, n_ext)
    w[[1, end], :] .= 0f0
    w[:, [1, end]] .= 0f0
    v = 1f0.-w
    v[:, [1, end]] = v[:, [1, end]]/h[2] # Sommerfeld radiation condition
    v[[1, end], :] = v[[1, end], :]/h[1] # Sommerfeld radiation condition
    # M = spdiagm(0 => vec(ω.^2f0.*w.*m_ext))+1im*spdiagm(0 => vec(ω.*v.*sqrt.(m_ext)))
    M = spdiagm(0 => vec(ω.^2f0.*w.*m_ext))+1im*spdiagm(0 => vec(ω.*v.*sqrt.(C.(m_ext))))

    # Assembly
    return -M-Δ

end

function taper(n::Tuple{Int64, Int64}, nabc::NTuple{4, Int64})

    taper_x = 1f0.-cat([range(-1f0, stop = 0f0, length = nabc[1])...].^2, zeros(R, n[1]), [range(0f0, stop = 1f0, length = nabc[2])...].^2, dims = 1)
    taper_z = 1f0.-cat([range(-1f0, stop = 0f0, length = nabc[3])...].^2, zeros(R, n[2]), [range(0f0, stop = 1f0, length = nabc[4])...].^2, dims = 1)
    return 1f0.-reshape(taper_x, :, 1)*reshape(taper_z, 1, :)

end

function computeNabc(h::NTuple{2, R}, m::Array{R, 2}, freq::R)

    return Int64.((round(1f0./(min(h...)*min(sqrt.(m[1, :])...)*freq)), round(1f0./(min(h...)*min(sqrt.(m[end, :])...)*freq)), round(1f0./(min(h...)*min(sqrt.(m[:, 1])...)*freq)), round(1f0./(min(h...)*min(sqrt.(m[:, end])...)*freq))))

end
