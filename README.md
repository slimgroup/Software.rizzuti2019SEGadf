# Software.rizzuti2019SEGadf - Frequency-domain implementation of the dual formulation of Wavefield Reconstruction Inversion

Julia implementation of the dual formulation of Wavefield Reconstruction Inversion (frequency domain)

Additional technical information about this work can be found in the SLIM group page:
https://slim.gatech.edu/content/dual-formulation-time-domain-wavefield-reconstruction-inversion-0

This software release complements the SEG San Antonio 2019 technical abstract:
https://library.seg.org/doi/10.1190/segam2019-3216760.1

## Requirements

Current supported Julia version: 1.2

Package dependencies (type in Julia REPL):
] add Optim
] add LineSearches
] add PyPlot
] add JLD2
] add ImageFiltering

## Instructions

To run the Gaussian lens inversion problem:

### Generate synthetic data

From the parent directory, type in Julia REPL:
include("./data/Gausslens/gendata_Gausslens.jl")

This will create a file "./data/Gausslens/Gausslens_data.jld" containing domain discretization details, true squared slowness model, source/receiver geometry, and frequency-domain synthetic data.

The squared slowness model can be inspected by typing:
imshow(m', cmap = "jet", aspect="auto")

Data in source/receiver coordinates can be inspected by typing:
imshow(real.(dat[1])', aspect = "auto", cmap = "jet")

### Run the WRI dual inversion script

From the parent directory, type in Julia REPL:
include("./scripts/inversion_Gausslens.jl")

In order to run FWI on the same model, comment/uncomment the relevant lines in the script.

## Author

Software written by Gabrio Rizzuti at Georgia Institute of Technology, rizzuti.gabrio@gatech.edu
