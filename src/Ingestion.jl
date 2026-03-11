module Ingestion

using PromptingTools
using StaticArrays
using Serialization

export Document, ParentChunk, ChildChunk, BinaryIndex, VectorDB
export load_documents, hierarchical_chunk, embed_and_quantize, save_database, load_database
export compress_to_binary

const SUPPORTED_EXTS = (".txt", ".md")

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

struct ChildChunk
    id::String
    parent_id::String
    text::String
end

"""In-memory binary index: vector of SVector{N,UInt8} for hamming search."""
struct BinaryIndex{N}
    vectors::Vector{SVector{N, UInt8}}  # heavily optimized zero-allocation distances
end

struct VectorDB{N}
    parents::Vector{ParentChunk}
    children::Vector{ChildChunk}
    index::BinaryIndex{N}
end

# ─── Document Loading ─────────────────────────────────────────────

function load_documents(folder_path::String)
    docs = Document[]
    if !isdir(folder_path)
        @warn "Directory $folder_path does not exist."
        return docs
    end
    for (root, dirs, files) in walkdir(folder_path)
        for file in files
            if any(ext -> endswith(lowercase(file), ext), SUPPORTED_EXTS)
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
    # Using SubStrings to avoid mass allocation
    matches = collect(eachmatch(r"\S+", text))
    chunks = SubString{String}[]
    step = max(1, chunk_size - overlap)
    i = 1
    n = length(matches)
    while i <= n
        start_idx = matches[i].offset
        end_idx = min(i + chunk_size - 1, n)
        # the end offset of the last word in the chunk
        end_offset = matches[end_idx].offset + length(matches[end_idx].match) - 1
        
        push!(chunks, SubString(text, start_idx, end_offset))
        i += step
    end
    return String.(chunks) # string conversion at the very end to free original string refs
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
    compress_to_binary(embedding::AbstractVector{<:AbstractFloat}) -> Vector{UInt8}

Generalized binary quantization: each float > 0 becomes a 1-bit,
packed 8 per byte. Handles any embedding dimension divisible by 8.
"""
function compress_to_binary(embedding::AbstractVector{F}) where {F<:AbstractFloat}
    dim = length(embedding)
    @assert dim % 8 == 0 "Embedding dimension ($dim) must be divisible by 8"
    
    n_bytes = dim ÷ 8
    out = Vector{UInt8}(undef, n_bytes)

    @inbounds for i in 0:(n_bytes-1)
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
        out[i+1] = byte_val
    end
    return out
end

# ─── Embedding + Quantization ────────────────────────────────────

function embed_and_quantize(children::Vector{ChildChunk}; model="embeddinggemma:300m-qat-q4_0")
    schema = PromptingTools.OllamaSchema()
    
    if isempty(children)
        return BinaryIndex{0}(SVector{0,UInt8}[])
    end
    
    @info "Embedding and quantizing $(length(children)) chunks..."
    
    # Process the first chunk to detect dimension
    msg1 = aiembed(schema, children[1].text; model=model)
    bin1 = compress_to_binary(msg1.content)
    N = length(bin1)
    
    binary_vectors = Vector{SVector{N, UInt8}}(undef, length(children))
    binary_vectors[1] = SVector{N, UInt8}(bin1)
    @info "Detected embedding dim=$(length(msg1.content)), quantized to $N bytes"
    
    if length(children) > 1
        asyncmap(2:length(children); ntasks=10) do i
            # Basic retry on embedding 
            local msg
            for attempt in 1:3
                try
                    msg = aiembed(schema, children[i].text; model=model)
                    break
                catch e
                    if attempt == 3
                        rethrow(e)
                    end
                    sleep(1)
                end
            end
            
            binary_vectors[i] = SVector{N, UInt8}(compress_to_binary(msg.content))
            if i % 10 == 0
                @info "Processed $i/$(length(children)) chunks."
            end
        end
    end
    @info "Done embedding and quantizing!"
    return BinaryIndex{N}(binary_vectors)
end

# ─── Persistence ──────────────────────────────────────────────────

function save_database(file_path::String, db::VectorDB{N}) where N
    serialize(file_path, db)
end

function load_database(file_path::String)
    return deserialize(file_path)
end

end # module
