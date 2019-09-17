################################################################################
#
# Weighting functions for PDE-misfit least-squares norm
#
################################################################################


export src_focusing_fun, src_focusing_mat, src_focusing_array, depth_weighting, depth_weighting_array, compute_srcfcsδ


function src_focusing_fun(q::Array{T, 2}, n::NTuple{2, Int64}, nabc::NTuple{4, Int64}, h::NTuple{2, R}, x_src::Array{R, 1}, z_src::R; δ::R = R(0.001)) where T <: RuC

    wq = Array{T}(undef, size(q))
    Ns = length(x_src)
    next = sz_abc(n, nabc)
    x = reshape(collect((0:next[1]-1)*h[1]).-nabc[1]*h[1], :, 1)
    z = reshape(collect((0:next[2]-1)*h[2]).-nabc[3]*h[2], 1, :)
    for idx_s = 1:Ns
        winv2 = ((x.-x_src[idx_s]).^R(2).+(z.-z_src).^R(2).+δ.^R(2))./δ.^R(2)
        wq[:, idx_s] = q[:, idx_s]./vec(winv2)
    end
    return wq

end

function src_focusing_mat(n::NTuple{2, Int64}, nabc::NTuple{4, Int64}, h::NTuple{2, R}, x_src::R, z_src::R; δ::R = R(0.001))

    next = sz_abc(n, nabc)
    x = reshape(collect((0:next[1]-1)*h[1]).-nabc[1]*h[1], :, 1)
    z = reshape(collect((0:next[2]-1)*h[2]).-nabc[3]*h[2], 1, :)
    return spdiagm(0 => vec(sqrt.(((x.-x_src).^R(2).+(z.-z_src).^R(2).+δ.^R(2))./δ.^R(2))))

end

function src_focusing_array(n::NTuple{2, Int64}, nabc::NTuple{4, Int64}, h::NTuple{2, R}, x_src::Array{R, 1}, z_src::R; δ::R = R(0.001))

    Ns = length(x_src)
    next = sz_abc(n, nabc)
    w = Array{R}(undef, (prod(next), Ns))
    x = reshape(collect((0:next[1]-1)*h[1]).-nabc[1]*h[1], :, 1)
    z = reshape(collect((0:next[2]-1)*h[2]).-nabc[3]*h[2], 1, :)
    for idx_s = 1:Ns
        w[:, idx_s] = vec(((x.-x_src[idx_s]).^R(2).+(z.-z_src).^R(2).+δ.^R(2))./δ.^R(2))
    end
    return w

end

function depth_weighting(q::Array{T, 2}, n::NTuple{2, Int64}, nabc::NTuple{4, Int64}, h::NTuple{2, R}, z_src::R; δ::R = 1f-4) where T <: RuC

    next = sz_abc(n, nabc)
    z = reshape(collect((0:next[2]-1)*h[2]).-nabc[3]*h[2], 1, :)
    winv2 = ((z.-z_src).^R(2).+δ.^R(2))./δ.^R(2)
    return reshape(reshape(q, (next..., size(q, 2)))./winv2, size(q))

end

function depth_weighting_array(n::NTuple{2, Int64}, nabc::NTuple{4, Int64}, h::NTuple{2, R}, z_src::R; δ::R = 1f-4)

    next = sz_abc(n, nabc)
    z = reshape(collect((0:next[2]-1)*h[2]).-nabc[3]*h[2], 1, :)
    return reshape(repeat(sqrt.(((z.-z_src).^2f0.+δ.^2f0)./δ.^2f0), outer = (next[1], 1)), :, 1)

end

function compute_srcfcsδ(m::R, freq::R)
    # Promote focusing on a "basin of attraction" corresponding to one wavelength in radius

    return sqrt(R(3)./m)./freq

end
