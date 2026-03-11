module Backend

using PromptingTools
using Base.Threads
using StaticArrays
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

using DataStructures

# ─── K-Closest Search ─────────────────────────────────────────────

function _k_closest(
    db::AbstractVector{V},
    query::AbstractVector{T},
    k::Int;
    startind::Int=1,
) where {T<:Integer,V<:AbstractVector{T}}
    # We use a MaxHeap to keep the k smallest distances (largest elements get popped).
    # Since DataStructures.BinaryMaxHeap stores elements and pops the maximum,
    # we want to keep the k items with the smallest distance.
    # We will use pairs, ordered by distance. Default order for Pair is to compare the first element,
    # which is exactly what we want (distance => index).
    heap = BinaryMaxHeap{Pair{Int,Int}}()
    sizehint!(heap, k + 1)
    
    @inbounds for i in eachindex(db)
        d = hamming_distance(db[i], query)
        push!(heap, d => startind + i - 1)
        if length(heap) > k
            pop!(heap) # remove the maximum distance
        end
    end
    return heap.valtree
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
    task_ranges = filter(!isempty, [(i:min(i + chunk_size - 1, n)) for i = 1:chunk_size:n])
    tasks = map(task_ranges) do r
        Threads.@spawn _k_closest(view(db, r), query, min(k, length(r)); startind=r[1])
    end
    results = fetch.(tasks)
    # Merge all heaps, sort, take top k (TODO: O(t*k*log(t)) merge ideally)
    merged = vcat(results...)
    sort!(merged, by=x -> x[1])
    return merged[1:min(k, length(merged))]
end

# ─── Retrieval ─────────────────────────────────────────────────────

function retrieve_context(
    query::String,
    db::VectorDB{N};
    k::Int=3,
    model="embeddinggemma:300m-qat-q4_0"
) where N
    # 1. Embed and quantize the query
    msg = aiembed(SCHEMA, query, copy; model=model)
    q_raw = Vector{Float32}(msg.content)
    q_bin = SVector{N, UInt8}(compress_to_binary(q_raw))

    # 2. Fast hamming search over quantized index
    db_vectors = db.index.vectors
    n_candidates = k * 2
    
    top_parent_ids = Set{String}()
    top_parents = ParentChunk[]
    
    # Pre-build dictionary for O(1) lookups
    parent_dict = Dict{String, ParentChunk}(p.id => p for p in db.parents)

    while length(top_parents) < k && n_candidates <= length(db_vectors)
        closest = k_closest_parallel(db_vectors, q_bin, n_candidates)
        
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

            if length(top_parents) >= k
                break
            end
        end
        
        if length(top_parents) >= k
            break
        end

        # Request a larger batch if we didn't get enough unique parents
        new_candidates = n_candidates * 4
        if n_candidates == length(db_vectors)
            break
        end
        n_candidates = min(length(db_vectors), new_candidates)
    end

    return top_parents
end

# ─── Generation ────────────────────────────────────────────────────

function generate_answer(query::String, context_parents::Vector{ParentChunk}; model="gemma3:1b")
    context_str = join([p.text for p in context_parents], "\n\n---\n\n")

    prompt = string("You are a helpful AI assistant. Use the following context to answer the user's question.\n",
        "If the answer is not contained in the context, say \"I don't have enough information to answer that.\"\n\n",
        "Context:\n", context_str, "\n\n",
        "User Question:\n", query)

    msg = aigenerate(SCHEMA, prompt; model=model)
    return msg.content
end

end # module
