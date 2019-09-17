push!(LOAD_PATH, string(pwd(), "/src/"))
using WRIdual
using LinearAlgebra, SparseArrays
using Optim, LineSearches
using PyPlot
using JLD2
using Dates


## Load slowness model and data

@load "./data/Gausslens/Gausslens_data.jld"


## Defining objective functional

# Background model
mb = 1e0./2000e0.^2e0.*ones(Float64, size(m))

# Setting model preprocessing and gradient preconditioning
mask = BitArray(undef, size(m)); mask .= false
mask[3:end-2, 3:end-2] .= true

# Source encoder
src_enc = C.(Matrix(sparse(I, length(x_src), length(x_src))))

# Pre-/post-processing for objective functionals
preproc(x) = contr2abs(x, mask, mb)
postproc(g) = gradprec_contr2abs(g, mask, mb)

# WRI dual functional setup
ε = 0.01
δ = 1e-1*compute_srcfcsδ(mb[1, 1], freqs[1])
weight_array = src_focusing_array(size(mb), nabc, h, x_src, z_src; δ = δ)
fun!(F, G, x) = objWRIdual_ε_yres!(F, G, preproc(x), h, [freqs[1]], x_rcv, z_rcv, x_src, z_src, src_enc, [dat[1]], ε; nabc = nabc, weight_array = weight_array, gradmprec_fun = postproc)

# # FWI functional setup
# fun!(F, G, x) = objFWI!(F, G, preproc(x), h, [freqs[1]], x_rcv, z_rcv, x_src, z_src, src_enc, [dat[1]]; nabc = nabc, gradprec_fun = postproc)


## Optimization parameters

# Starting guess
x0 = zeros(Float64, length(findall(mask)))

# Options
# method = LBFGS(; alphaguess = InitialStatic(; alpha = 1f4), linesearch = BackTracking())
method = LBFGS()
niter = 20
opt = Optim.Options(iterations = niter, store_trace = true, show_trace  = true, show_every  = 1)


## Run optimization

# Minimization
result = optimize(Optim.only_fg!(fun!), x0, method, opt)
x0 = Optim.minimizer(result)
minv = preproc(x0)
loss_log = Optim.f_trace(result)

# Plot result
imshow(minv', aspect = "auto", cmap = "jet")
