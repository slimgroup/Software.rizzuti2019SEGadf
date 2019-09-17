################################################################################
#
# Restriction operator linear algebra routines
#
################################################################################


## Restriction operator (explicit matrix)

export restrOpMat


function restrOpMat(n::NTuple{2, Int64}, h::NTuple{2, R}, xint::Array{R, 1}, zint::R, nabc::NTuple{4, Int64})

    # Neighboring indexes for bilinear interp
    idx_x = xint./h[1].+R(1)
    idx_x1 = floor.(idx_x)
    idx_x2 = idx_x1.+R(1)
    idx_z = zint./h[2].+R(1)
    idx_z1 = floor.(idx_z)
    idx_z2 = idx_z1.+R(1)

    j_11 = Int64.([idx_x1...]).+nabc[1].+(Int64(idx_z1)+nabc[3]-1)*(n[1]+nabc[1]+nabc[2])
    j_12 = Int64.([idx_x1...]).+nabc[1].+(Int64(idx_z2)+nabc[3]-1)*(n[1]+nabc[1]+nabc[2])
    j_21 = Int64.([idx_x2...]).+nabc[1].+(Int64(idx_z1)+nabc[3]-1)*(n[1]+nabc[1]+nabc[2])
    j_22 = Int64.([idx_x2...]).+nabc[1].+(Int64(idx_z2)+nabc[3]-1)*(n[1]+nabc[1]+nabc[2])

    # Weights
    v_11 = (1f0.-abs.(idx_x.-idx_x1)).*(1f0.-abs.(idx_z-idx_z1))
    v_12 = (1f0.-abs.(idx_x.-idx_x1)).*(1f0.-abs.(idx_z-idx_z2))
    v_21 = (1f0.-abs.(idx_x.-idx_x2)).*(1f0.-abs.(idx_z-idx_z1))
    v_22 = (1f0.-abs.(idx_x.-idx_x2)).*(1f0.-abs.(idx_z-idx_z2))

    # Assembly
    nx = length(xint)
    i = collect(1:nx)
    next = prod(sz_abc(n, nabc))
    return sparse(i, j_11, v_11, nx, next)+
           sparse(i, j_12, v_12, nx, next)+
           sparse(i, j_21, v_21, nx, next)+
           sparse(i, j_22, v_22, nx, next)

end
