module Cerebro

include("Ingestion.jl")
include("Backend.jl")
include("App.jl")

using .Ingestion
using .Backend
using .App

export Ingestion, Backend, App

end # module Cerebro
