using Test
using StaticArrays

# We need to include the modules directly for testing in isolation
# since they are submodules of Cerebro

include("test_config.jl")
include("test_ingestion.jl")
include("test_backend.jl")
