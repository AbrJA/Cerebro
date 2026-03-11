module Backend

using PromptingTools
using Base.Threads
using StaticArrays
using ..Config: CerebroConfig
using ..Ingestion: ParentChunk, ChildChunk, BinaryIndex, compress_to_binary, VectorDB

export retrieve_context, generate_answer

const SCHEMA = PromptingTools.OllamaSchema()

# ─── SIMD Hamming Distance ────────────────────────────────────────

@inline hamming_distance(x1::T, x2::T) where T<:Integer = count_ones(x1 ⊻ x2)

function hamming_distance(x1, x2)
    s = 0
    @inbounds @simd for i in eachindex(x1, x2)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

# ─── Zero-Allocation MaxHeap ─────────────────────────────────────

"""
    MaxHeap(k)

Fixed-size max-heap for top-k smallest distance selection.
Pre-allocates a backing array of size k — zero allocation after construction.
"""
mutable struct MaxHeap
    const data::Vector{Pair{Int,Int}}   # distance => index
    current_idx::Int                     # fill pointer
    const k::Int

    function MaxHeap(k::Int)
        new(fill((typemax(Int) => -1), k), 1, k)
    end
end

"""Insert a (distance => index) pair. After k insertions, only keeps the k smallest distances."""
function heap_push!(heap::MaxHeap, value::Pair{Int,Int})
    if heap.current_idx <= heap.k
        # Phase 1: fill the backing array
        heap.data[heap.current_idx] = value
        heap.current_idx += 1
        if heap.current_idx > heap.k
            _makeheap!(heap)
        end
    elseif value.first < heap.data[1].first
        # Phase 2: replace root if new value is smaller
        heap.data[1] = value
        _heapify!(heap, 1)
    end
end

function _makeheap!(heap::MaxHeap)
    for i in div(heap.k, 2):-1:1
        _heapify!(heap, i)
    end
end

"""Iterative max-heapify (no stack overflow risk)."""
function _heapify!(heap::MaxHeap, i::Int)
    n = length(heap.data)
    while true
        left = 2 * i
        right = 2 * i + 1
        largest = i

        if left <= n && heap.data[left].first > heap.data[largest].first
            largest = left
        end
        if right <= n && heap.data[right].first > heap.data[largest].first
            largest = right
        end
        if largest == i
            break
        end
        heap.data[i], heap.data[largest] = heap.data[largest], heap.data[i]
        i = largest
    end
end

# ─── K-Closest Search ─────────────────────────────────────────────

function _k_closest(
    db::AbstractVector{V},
    query::AbstractVector{T},
    k::Int;
    startind::Int=1,
) where {T<:Integer,V<:AbstractVector{T}}
    heap = MaxHeap(k)
    @inbounds for i in eachindex(db)
        d = hamming_distance(db[i], query)
        heap_push!(heap, d => startind + i - 1)
    end
    return heap.data
end

function k_closest_parallel(
    db::AbstractVector{V},
    query::AbstractVector{T},
    k::Int;
    min_parallel_threshold::Int=1000,
) where {T<:Integer,V<:AbstractVector{T}}
    n = length(db)

    # Skip parallelism for small databases — thread spawn overhead dominates
    if n < min_parallel_threshold
        result = _k_closest(db, query, k)
        sort!(result; by=x -> x[1])
        return result[1:min(k, length(result))]
    end

    t = nthreads()
    chunk_size = max(1, n ÷ t)
    task_ranges = filter(!isempty, [(i:min(i + chunk_size - 1, n)) for i = 1:chunk_size:n])

    tasks = map(task_ranges) do r
        Threads.@spawn _k_closest(view(db, r), query, min(k, length(r)); startind=r[1])
    end
    results = fetch.(tasks)

    # Pre-allocated merge instead of vcat
    total_len = sum(length, results)
    merged = Vector{Pair{Int,Int}}(undef, total_len)
    offset = 1
    for r in results
        copyto!(merged, offset, r, 1, length(r))
        offset += length(r)
    end

    # Final top-k selection via MaxHeap (more efficient than sort for large t*k)
    final_heap = MaxHeap(k)
    for pair in merged
        pair.second >= 1 && heap_push!(final_heap, pair)
    end

    result = filter(p -> p.second >= 1, final_heap.data)
    sort!(result; by=x -> x[1])
    return result[1:min(k, length(result))]
end

# ─── Retrieval ─────────────────────────────────────────────────────

function retrieve_context(
    query::AbstractString,
    db::VectorDB{N};
    config::CerebroConfig=CerebroConfig()
) where N
    k = config.retrieval_k

    # 1. Embed and quantize the query
    msg = aiembed(SCHEMA, query; model=config.embedding_model)
    q_raw = Vector{Float32}(msg.content)
    q_bin = SVector{N,UInt8}(compress_to_binary(q_raw))

    # 2. Fast hamming search over quantized index
    db_vectors = db.index.vectors
    n_candidates = k * 2

    top_parent_ids = Set{String}()
    top_parents = ParentChunk[]

    # Use pre-built lookup from VectorDB (O(1) per parent)
    parent_dict = db.parent_lookup

    while length(top_parents) < k && n_candidates <= length(db_vectors)
        closest = k_closest_parallel(db_vectors, q_bin, n_candidates;
            min_parallel_threshold=config.min_parallel_threshold)

        empty!(top_parent_ids)
        empty!(top_parents)

        for pair in closest
            child_idx = pair.second
            if child_idx < 1 || child_idx > length(db.children)
                continue
            end
            p_id = db.children[child_idx].parent_id

            if !(p_id in top_parent_ids)
                push!(top_parent_ids, p_id)
                if haskey(parent_dict, p_id)
                    push!(top_parents, parent_dict[p_id])
                end
            end

            length(top_parents) >= k && break
        end

        length(top_parents) >= k && break

        # Request a larger batch if we didn't get enough unique parents
        new_candidates = n_candidates * 4
        n_candidates == length(db_vectors) && break
        n_candidates = min(length(db_vectors), new_candidates)
    end

    return top_parents
end

# ─── Generation ────────────────────────────────────────────────────

"""
    generate_answer(query, context_parents; config=CerebroConfig())

Generate an answer using structured messages (SystemMessage + UserMessage)
with context-length truncation.
"""
function generate_answer(query::AbstractString, context_parents::Vector{ParentChunk}; config::CerebroConfig=CerebroConfig())
    # Context truncation: limit total words to prevent exceeding model context window
    context_parts = String[]
    total_words = 0
    for p in context_parents
        words_in_chunk = length(split(p.text))
        if total_words + words_in_chunk > config.max_context_words
            # Truncate this chunk to fit
            remaining = config.max_context_words - total_words
            if remaining > 0
                truncated = join(split(p.text)[1:min(remaining, words_in_chunk)], " ")
                push!(context_parts, truncated)
            end
            break
        end
        push!(context_parts, p.text)
        total_words += words_in_chunk
    end

    context_str = join(context_parts, "\n\n---\n\n")

    # Use structured messages for proper model separation
    sys_msg = PromptingTools.SystemMessage(
        "You are a helpful AI assistant. Use the following context to answer the user's question. " *
        "If the answer is not contained in the context, say \"I don't have enough information to answer that.\""
    )

    user_msg = PromptingTools.UserMessage(
        string("Context:\n", context_str, "\n\nQuestion:\n", query)
    )

    msg = aigenerate(SCHEMA, [sys_msg, user_msg]; model=config.chat_model)
    return msg.content
end

end # module Backend
