module Cerebro

include("Ingestion.jl")
include("Backend.jl")
include("App.jl")

using .Ingestion
using .Backend
using .App

export retrieve_context, generate_answer, start_server, load_documents, hierarchical_chunk, embed_and_quantize, save_database, load_database, Document, ParentChunk, ChildChunk, BinaryIndex, VectorDB, compress_to_binary

function __init__()
    @info "Cerebro RAG module successfully loaded."
end

end # module Cerebro
