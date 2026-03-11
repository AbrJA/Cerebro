module Ingestion

using PromptingTools
using StaticArrays
using JSON3
using StructTypes

export Document, ParentChunk, ChildChunk, BinaryIndex
export load_documents, hierarchical_chunk, embed_and_quantize!, save_database, load_database
export compress_to_binary

# ─── Data Structures ──────────────────────────────────────────────

struct Document
    id::String
    content::String
end

struct ParentChunk
    id::String
    doc_id::String
    text::String
end
StructTypes.StructType(::Type{ParentChunk}) = StructTypes.Struct()

struct ChildChunk
    id::String
    parent_id::String
    text::String
end
StructTypes.StructType(::Type{ChildChunk}) = StructTypes.Struct()

"""In-memory binary index: vector of SVector{N,Int8} for hamming search."""
struct BinaryIndex
    vectors::Vector{Vector{Int8}}  # each inner vector is the binary-quantized embedding
    n_bytes::Int                   # number of bytes per quantized vector
end
StructTypes.StructType(::Type{BinaryIndex}) = StructTypes.Struct()

struct VectorDB
    parents::Vector{ParentChunk}
    children::Vector{ChildChunk}
    index::BinaryIndex
end
StructTypes.StructType(::Type{VectorDB}) = StructTypes.Struct()

# ─── Document Loading ─────────────────────────────────────────────

function load_documents(folder_path::String)
    docs = Document[]
    if !isdir(folder_path)
        @warn "Directory $folder_path does not exist."
        return docs
    end
    for (root, dirs, files) in walkdir(folder_path)
        for file in files
            if endswith(lowercase(file), ".txt") || endswith(lowercase(file), ".md")
                path = joinpath(root, file)
                content = read(path, String)
                push!(docs, Document(path, content))
            end
        end
    end
    return docs
end

# ─── Chunking ─────────────────────────────────────────────────────

function _chunk_text(text::String, chunk_size::Int, overlap::Int)
    words = split(text)
    chunks = String[]
    step = max(1, chunk_size - overlap)
    i = 1
    while i <= length(words)
        end_idx = min(i + chunk_size - 1, length(words))
        push!(chunks, join(words[i:end_idx], " "))
        i += step
    end
    return chunks
end

function hierarchical_chunk(docs::Vector{Document}; parent_words=300, child_words=75, parent_overlap=50, child_overlap=15)
    parents = ParentChunk[]
    children = ChildChunk[]

    for doc in docs
        parent_texts = _chunk_text(doc.content, parent_words, parent_overlap)
        for (i, p_text) in enumerate(parent_texts)
            p_id = "$(doc.id)_p$i"
            push!(parents, ParentChunk(p_id, doc.id, p_text))

            child_texts = _chunk_text(p_text, child_words, child_overlap)
            for (j, c_text) in enumerate(child_texts)
                c_id = "$(p_id)_c$j"
                push!(children, ChildChunk(c_id, p_id, c_text))
            end
        end
    end
    return parents, children
end

# ─── Binary Quantization ─────────────────────────────────────────

"""
    compress_to_binary(embedding::AbstractVector{<:AbstractFloat}) -> Vector{Int8}

Generalized binary quantization: each float > 0 becomes a 1-bit,
packed 8 per byte. Handles any embedding dimension divisible by 8.
"""
function compress_to_binary(embedding::AbstractVector{F}) where {F<:AbstractFloat}
    dim = length(embedding)
    n_bytes = dim ÷ 8
    out = Vector{Int8}(undef, n_bytes)

    @inbounds for i in 0:(n_bytes - 1)
        byte_val = 0x00
        base_idx = i * 8
        byte_val |= (UInt8(embedding[base_idx+1] > 0.0f0) << 0)
        byte_val |= (UInt8(embedding[base_idx+2] > 0.0f0) << 1)
        byte_val |= (UInt8(embedding[base_idx+3] > 0.0f0) << 2)
        byte_val |= (UInt8(embedding[base_idx+4] > 0.0f0) << 3)
        byte_val |= (UInt8(embedding[base_idx+5] > 0.0f0) << 4)
        byte_val |= (UInt8(embedding[base_idx+6] > 0.0f0) << 5)
        byte_val |= (UInt8(embedding[base_idx+7] > 0.0f0) << 6)
        byte_val |= (UInt8(embedding[base_idx+8] > 0.0f0) << 7)
        out[i+1] = reinterpret(Int8, byte_val)
    end
    return out
end

# ─── Embedding + Quantization ────────────────────────────────────

function embed_and_quantize!(children::Vector{ChildChunk}; model="embeddinggemma:300m-qat-q4_0")
    schema = PromptingTools.OllamaSchema()
    binary_vectors = Vector{Vector{Int8}}()
    n_bytes = 0

    println("Embedding and quantizing $(length(children)) chunks...")
    for i in eachindex(children)
        msg = aiembed(schema, children[i].text, copy; model=model)
        raw = Vector{Float32}(msg.content)

        bin = compress_to_binary(raw)
        push!(binary_vectors, bin)

        if n_bytes == 0
            n_bytes = length(bin)
            println("  Detected embedding dim=$(length(raw)), quantized to $n_bytes bytes")
        end

        if i % 10 == 0
            println("  Processed $i/$(length(children)) chunks.")
        end
    end
    println("Done embedding and quantizing!")
    return BinaryIndex(binary_vectors, n_bytes)
end

# ─── Persistence ──────────────────────────────────────────────────

function save_database(file_path::String, parents::Vector{ParentChunk}, children::Vector{ChildChunk}, index::BinaryIndex)
    db = VectorDB(parents, children, index)
    open(file_path, "w") do f
        JSON3.write(f, db)
    end
end

function load_database(file_path::String)
    bytes = read(file_path)
    db = JSON3.read(bytes, VectorDB)
    return db.parents, db.children, db.index
end

end # module
