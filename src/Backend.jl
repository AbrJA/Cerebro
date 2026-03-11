module Backend

using PromptingTools
using Base.Threads
using ..Ingestion: ParentChunk, ChildChunk, BinaryIndex, compress_to_binary

export retrieve_context, generate_answer

# ─── SIMD Hamming Distance ────────────────────────────────────────

@inline hamming_distance(x1::T, x2::T) where T<:Integer = count_ones(x1 ⊻ x2)

function hamming_distance(x1, x2)
    s = 0
    @inbounds @simd for i in eachindex(x1, x2)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

# ─── MaxHeap for Top-K (zero-allocation after init) ──────────────

mutable struct MaxHeap
    const data::Vector{Pair{Int,Int}}
    current_idx::Int
    const k::Int

    function MaxHeap(k::Int)
        new(fill((typemax(Int) => -1), k), 1, k)
    end
end

function insert!(heap::MaxHeap, value::Pair{Int,Int})
    if heap.current_idx <= heap.k
        heap.data[heap.current_idx] = value
        heap.current_idx += 1
        if heap.current_idx > heap.k
            makeheap!(heap)
        end
    elseif value.first < heap.data[1].first
        heap.data[1] = value
        heapify!(heap, 1)
    end
end

function makeheap!(heap::MaxHeap)
    for i in div(heap.k, 2):-1:1
        heapify!(heap, i)
    end
end

function heapify!(heap::MaxHeap, i::Int)
    left = 2 * i
    right = 2 * i + 1
    largest = i

    if left <= length(heap.data) && heap.data[left].first > heap.data[largest].first
        largest = left
    end

    if right <= length(heap.data) && heap.data[right].first > heap.data[largest].first
        largest = right
    end

    if largest != i
        heap.data[i], heap.data[largest] = heap.data[largest], heap.data[i]
        heapify!(heap, largest)
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
        insert!(heap, d => startind + i - 1)
    end
    return heap.data
end

function k_closest_parallel(
    db::AbstractVector{V},
    query::AbstractVector{T},
    k::Int,
) where {T<:Integer,V<:AbstractVector{T}}
    n = length(db)
    t = nthreads()
    # Ensure we don't create empty ranges
    chunk_size = max(1, n ÷ t)
    task_ranges = [(i:min(i + chunk_size - 1, n)) for i = 1:chunk_size:n]
    tasks = map(task_ranges) do r
        Threads.@spawn _k_closest(view(db, r), query, min(k, length(r)); startind=r[1])
    end
    results = fetch.(tasks)
    # Merge all heaps, sort, take top k
    merged = vcat(results...)
    sort!(merged, by=x -> x[1])
    return merged[1:min(k, length(merged))]
end

# ─── Retrieval ─────────────────────────────────────────────────────

function retrieve_context(
    query::String,
    parents::Vector{ParentChunk},
    children::Vector{ChildChunk},
    index::BinaryIndex;
    k::Int=3,
    model="embeddinggemma:300m-qat-q4_0"
)
    schema = PromptingTools.OllamaSchema()

    # 1. Embed and quantize the query
    msg = aiembed(schema, query, copy; model=model)
    q_raw = Vector{Float32}(msg.content)
    q_bin = compress_to_binary(q_raw)

    # 2. Fast hamming search over quantized index
    db = index.vectors
    # Fetch more candidates than k to ensure we get k unique parents
    n_candidates = min(length(db), k * 5)
    closest = k_closest_parallel(db, q_bin, n_candidates)

    # 3. Map child indices → unique parent chunks
    top_parent_ids = String[]
    top_parents = ParentChunk[]

    for pair in closest
        child_idx = pair.second
        if child_idx < 1 || child_idx > length(children)
            continue
        end
        p_id = children[child_idx].parent_id

        if !(p_id in top_parent_ids)
            push!(top_parent_ids, p_id)
            p_idx = findfirst(p -> p.id == p_id, parents)
            if p_idx !== nothing
                push!(top_parents, parents[p_idx])
            end
        end

        if length(top_parents) >= k
            break
        end
    end

    return top_parents
end

# ─── Generation ────────────────────────────────────────────────────

function generate_answer(query::String, context_parents::Vector{ParentChunk}; model="gemma3:1b")
    context_str = join([p.text for p in context_parents], "\n\n---\n\n")

    prompt = """
    You are a helpful AI assistant. Use the following context to answer the user's question.
    If the answer is not contained in the context, say "I don't have enough information to answer that."

    Context:
    $context_str

    User Question:
    $query
    """

    schema = PromptingTools.OllamaSchema()
    msg = aigenerate(schema, prompt; model=model)
    return msg.content
end

end # module
