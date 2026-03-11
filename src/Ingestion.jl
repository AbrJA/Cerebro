module Ingestion

using PromptingTools
using StaticArrays
using JLD2
using Serialization  # backwards-compat loading only

using ..Config: CerebroConfig

export Document, ParentChunk, ChildChunk, BinaryIndex, VectorDB
export load_documents, hierarchical_chunk, embed_and_quantize, save_database, load_database
export compress_to_binary, compress_to_binary!

const SUPPORTED_EXTS = (".txt", ".md")
const DB_FORMAT_VERSION = 1
const DB_FORMAT_MARKER = :cerebro_db_v2

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
    vectors::Vector{SVector{N, UInt8}}
end

"""
    VectorDB{N}

The core vector database holding parents, children, binary index,
and a pre-built parent lookup dictionary for O(1) access.
"""
struct VectorDB{N}
    parents::Vector{ParentChunk}
    children::Vector{ChildChunk}
    index::BinaryIndex{N}
    parent_lookup::Dict{String, ParentChunk}
end

# Convenience constructor that auto-builds the lookup dict
function VectorDB(parents::Vector{ParentChunk}, children::Vector{ChildChunk}, index::BinaryIndex{N}) where N
    lookup = Dict{String, ParentChunk}(p.id => p for p in parents)
    VectorDB{N}(parents, children, index, lookup)
end

# ─── Document Loading ─────────────────────────────────────────────

"""
    load_documents(folder_path; max_file_bytes=50_000_000)

Load all supported text files from `folder_path` recursively.
Deduplicates by inode, skips oversized files, and validates UTF-8 encoding.
"""
function load_documents(folder_path::String; max_file_bytes::Int=50_000_000)
    docs = Document[]
    if !isdir(folder_path)
        @warn "Directory does not exist" folder_path
        return docs
    end

    seen_inodes = Set{UInt64}()

    for (root, _, files) in walkdir(folder_path)
        for file in files
            any(ext -> endswith(lowercase(file), ext), SUPPORTED_EXTS) || continue

            path = joinpath(root, file)
            s = stat(path)

            # Deduplicate by inode (catches symlinks and hardlinks)
            s.inode in seen_inodes && continue
            push!(seen_inodes, s.inode)

            # File-size guard
            if s.size > max_file_bytes
                @warn "Skipping oversized file" path size_mb=round(s.size / 1_000_000; digits=1)
                continue
            end

            # Encoding validation
            content = try
                read(path, String)
            catch e
                @warn "Could not read file (encoding error?)" path exception=e
                continue
            end

            # Use relative path as ID for portability
            doc_id = relpath(path, folder_path)
            push!(docs, Document(doc_id, content))
        end
    end

    @info "Loaded $(length(docs)) documents from $folder_path"
    return docs
end

# ─── Chunking ─────────────────────────────────────────────────────

"""
    _chunk_text(text, chunk_size, overlap)

Word-level chunking using SubString views for minimal allocation.
"""
function _chunk_text(text::String, chunk_size::Int, overlap::Int)
    matches = collect(eachmatch(r"\S+", text))
    isempty(matches) && return String[]
    
    chunks = SubString{String}[]
    step = max(1, chunk_size - overlap)
    i = 1
    n = length(matches)

    while i <= n
        start_idx = matches[i].offset
        end_idx = min(i + chunk_size - 1, n)
        end_offset = matches[end_idx].offset + length(matches[end_idx].match) - 1

        push!(chunks, SubString(text, start_idx, end_offset))
        i += step
    end

    return String.(chunks)
end

"""
    _chunk_text_structured(text, chunk_size, overlap)

Paragraph-aware chunking: splits on paragraph breaks first,
then word-chunks within, preserving document structure.
"""
function _chunk_text_structured(text::String, chunk_size::Int, overlap::Int)
    paragraphs = split(text, r"\n{2,}")
    chunks = String[]
    buffer_words = String[]
    
    for para in paragraphs
        words = split(strip(para))
        isempty(words) && continue
        
        if length(buffer_words) + length(words) <= chunk_size
            append!(buffer_words, words)
        else
            # Flush current buffer
            if !isempty(buffer_words)
                push!(chunks, join(buffer_words, " "))
            end
            
            # If this paragraph alone exceeds chunk_size, word-chunk it
            if length(words) > chunk_size
                step = max(1, chunk_size - overlap)
                wi = 1
                while wi <= length(words)
                    chunk_end = min(wi + chunk_size - 1, length(words))
                    push!(chunks, join(words[wi:chunk_end], " "))
                    wi += step
                end
                buffer_words = String[]
            else
                buffer_words = collect(words)
            end
        end
    end
    
    !isempty(buffer_words) && push!(chunks, join(buffer_words, " "))
    return chunks
end

function hierarchical_chunk(docs::Vector{Document}; 
                           parent_words=300, child_words=75, 
                           parent_overlap=50, child_overlap=15,
                           structured=true)
    parents = ParentChunk[]
    children = ChildChunk[]
    
    chunk_fn = structured ? _chunk_text_structured : _chunk_text

    for doc in docs
        parent_texts = chunk_fn(doc.content, parent_words, parent_overlap)
        for (i, p_text) in enumerate(parent_texts)
            p_id = "$(doc.id)_p$i"
            push!(parents, ParentChunk(p_id, doc.id, p_text))

            child_texts = chunk_fn(p_text, child_words, child_overlap)
            for (j, c_text) in enumerate(child_texts)
                c_id = "$(p_id)_c$j"
                push!(children, ChildChunk(c_id, p_id, c_text))
            end
        end
    end

    @info "Chunking complete" n_parents=length(parents) n_children=length(children)
    return parents, children
end

# ─── Binary Quantization ─────────────────────────────────────────

"""
    compress_to_binary(embedding) -> Vector{UInt8}

Binary quantization: each float > 0 becomes a 1-bit, packed 8 per byte.
"""
function compress_to_binary(embedding::AbstractVector{F}) where {F<:AbstractFloat}
    dim = length(embedding)
    dim % 8 == 0 || throw(ArgumentError("Embedding dimension ($dim) must be divisible by 8"))

    n_bytes = dim ÷ 8
    out = Vector{UInt8}(undef, n_bytes)
    compress_to_binary!(out, embedding)
    return out
end

"""
    compress_to_binary!(out, embedding) -> out

In-place binary quantization into pre-allocated buffer. Zero-allocation hot path.
"""
function compress_to_binary!(out::Vector{UInt8}, embedding::AbstractVector{F}) where {F<:AbstractFloat}
    dim = length(embedding)
    n_bytes = dim ÷ 8
    @boundscheck begin
        dim % 8 == 0 || throw(ArgumentError("Embedding dimension ($dim) must be divisible by 8"))
        length(out) >= n_bytes || throw(ArgumentError("Output buffer too small: $(length(out)) < $n_bytes"))
    end

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

"""
    embed_and_quantize(children; config=CerebroConfig())

Embed and binary-quantize all child chunks. Uses concurrent async tasks
with exponential backoff retry and progress reporting.
"""
function embed_and_quantize(children::Vector{ChildChunk}; config::CerebroConfig=CerebroConfig())
    schema = PromptingTools.OllamaSchema()
    model = config.embedding_model

    if isempty(children)
        return BinaryIndex{0}(SVector{0,UInt8}[])
    end

    @info "Embedding and quantizing $(length(children)) chunks..."

    # Process the first chunk to detect dimension
    msg1 = _embed_with_retry(schema, children[1].text, model; max_retries=config.embedding_max_retries)
    bin1 = compress_to_binary(msg1.content)
    N = length(bin1)

    binary_vectors = Vector{SVector{N, UInt8}}(undef, length(children))
    binary_vectors[1] = SVector{N, UInt8}(bin1)
    @info "Detected embedding dim=$(length(msg1.content)), quantized to $N bytes"

    # Pre-allocate a reusable buffer per task (one per ntask)
    if length(children) > 1
        total = length(children)
        processed = Threads.Atomic{Int}(1)

        asyncmap(2:total; ntasks=config.embedding_ntasks) do i
            msg = _embed_with_retry(schema, children[i].text, model; max_retries=config.embedding_max_retries)
            binary_vectors[i] = SVector{N, UInt8}(compress_to_binary(msg.content))
            
            count = Threads.atomic_add!(processed, 1) + 1
            if count % 25 == 0 || count == total
                @info "Embedded $count/$total chunks"
            end
        end
    end

    @info "Done embedding and quantizing!"
    return BinaryIndex{N}(binary_vectors)
end

"""Embed with exponential backoff retry."""
function _embed_with_retry(schema, text::String, model::String; max_retries::Int=3)
    local msg
    for attempt in 1:max_retries
        try
            msg = aiembed(schema, text; model=model)
            return msg
        catch e
            if attempt == max_retries
                @error "Embedding failed after $max_retries attempts" exception=e
                rethrow(e)
            end
            wait_time = 2^attempt  # exponential backoff: 2, 4, 8 seconds
            @warn "Embedding attempt $attempt failed, retrying in $(wait_time)s..." exception=e
            sleep(wait_time)
        end
    end
end

# ─── Persistence (JLD2 with atomic writes) ───────────────────────

"""
    save_database(file_path, db)

Save VectorDB to disk using JLD2 with atomic write (write to temp, then rename).
Includes format marker and version for forward compatibility.
"""
function save_database(file_path::String, db::VectorDB{N}) where N
    tmp = file_path * ".tmp.$(getpid())"
    try
        jldsave(tmp;
            format_marker = DB_FORMAT_MARKER,
            format_version = DB_FORMAT_VERSION,
            parents = db.parents,
            children = db.children,
            binary_vectors = db.index.vectors,
            N = N
        )
        mv(tmp, file_path; force=true)
        @info "Database saved" path=file_path size_kb=round(filesize(file_path) / 1024; digits=1)
    catch e
        isfile(tmp) && rm(tmp)
        rethrow(e)
    end
end

"""
    load_database(file_path)

Load VectorDB from disk. Supports JLD2 format (v2) with fallback to legacy Serialization format.
"""
function load_database(file_path::String)
    if !isfile(file_path)
        throw(ArgumentError("Database file not found: $file_path"))
    end

    # Try JLD2 first
    try
        data = jldopen(file_path, "r") do f
            marker = f["format_marker"]
            version = f["format_version"]
            if marker !== DB_FORMAT_MARKER
                @warn "Unknown format marker" marker
            end
            if version != DB_FORMAT_VERSION
                @warn "Database version mismatch" expected=DB_FORMAT_VERSION got=version
            end
            parents = f["parents"]
            children = f["children"]
            binary_vectors = f["binary_vectors"]
            N_val = f["N"]
            return VectorDB(parents, children, BinaryIndex{N_val}(binary_vectors))
        end
        @info "Database loaded (JLD2)" path=file_path
        return data
    catch e
        @info "JLD2 load failed, trying legacy Serialization format..." exception=e
    end

    # Fallback: legacy Serialization format
    try
        data = deserialize(file_path)
        @info "Database loaded (legacy format)" path=file_path
        # If it's already a VectorDB but without parent_lookup, reconstruct
        if hasproperty(data, :parents) && hasproperty(data, :children) && hasproperty(data, :index)
            return VectorDB(data.parents, data.children, data.index)
        end
        return data
    catch e2
        @error "Failed to load database in any format" exception=e2
        rethrow(e2)
    end
end

end # module Ingestion
