module Cerebro

include("Config.jl")
include("Ingestion.jl")
include("Backend.jl")
include("App.jl")

using .Config
using .Ingestion
using .Backend
using .App

# Re-export key public API
export CerebroConfig
export load_documents, hierarchical_chunk, embed_and_quantize
export save_database, load_database
export compress_to_binary, compress_to_binary!
export Document, ParentChunk, ChildChunk, BinaryIndex, VectorDB
export retrieve_context, generate_answer
export start_server

function __init__()
    @info "Cerebro RAG v0.2.0 loaded"
    @info "  Julia $(VERSION) | Threads: $(Threads.nthreads())"
end

end # module Cerebro
