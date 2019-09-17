push!(LOAD_PATH, string(pwd(), "/src/"))
using WRIdual
using LinearAlgebra, SparseArrays
using PyPlot
using JLD2


## Set slowness model and discretization step

n = (201, 201)
h = (10e0, 10e0)
val = 0.5e0
x = reshape((collect(1:n[1]).-1f0).*h[1], :, 1)
z = reshape((collect(1:n[2]).-1f0).*h[2], 1, :)
v = 1000e0.*(2e0.-val*exp.(-((x./1000e0.-1e0)/0.25e0).^2e0.-((z./1000e0.-1e0)/0.5e0).^2e0))
m = 1e0./v.^2e0


## Create data

# Set frequencies
freqs = [6e0]

# Set source/receiver geometry
ix_src = 3:4:199
Ns = length(ix_src)
iz_src = 2
x_src = (Float64.(ix_src).-1e0)*h[1]
z_src = (Float64.(iz_src).-1e0)*h[2]
ix_rcv = 3:199
iz_rcv = 200
x_rcv = (Float64.(ix_rcv).-1e0)*h[1]
z_rcv = (Float64.(iz_rcv).-1e0)*h[2]

# Setting source wavelet
src_enc = C.(Matrix(sparse(I, length(x_src), length(x_src))))

# Absorbing boundary condition padding
nabc = (33, 33, 33, 33)

# Solving PDE for each frequency
dat = gendata(m, h, freqs, x_rcv, z_rcv, x_src, z_src, src_enc; nabc = nabc, verbose = true)


## Save data & related parameters

@save "./data/Gausslens/Gausslens_data.jld" h m freqs x_rcv z_rcv x_src z_src src_enc nabc dat
