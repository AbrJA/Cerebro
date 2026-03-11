module Config

export CerebroConfig

"""
    CerebroConfig

Centralized configuration for the entire Cerebro RAG pipeline.
Replaces scattered magic defaults across all modules.
"""
Base.@kwdef struct CerebroConfig
    # ── Embedding ──────────────────────────────────────────────────
    embedding_model::String     = "embeddinggemma:300m-qat-q4_0"
    embedding_batch_size::Int   = 32
    embedding_ntasks::Int       = 10
    embedding_max_retries::Int  = 3
    embedding_timeout_s::Int    = 30

    # ── Chunking ───────────────────────────────────────────────────
    parent_words::Int    = 300
    child_words::Int     = 75
    parent_overlap::Int  = 50
    child_overlap::Int   = 15

    # ── Search ─────────────────────────────────────────────────────
    retrieval_k::Int             = 3
    min_parallel_threshold::Int  = 1000   # skip parallelism below this

    # ── Generation ─────────────────────────────────────────────────
    chat_model::String       = "gemma3:1b"
    max_context_words::Int   = 3000   # truncate context beyond this

    # ── Server ─────────────────────────────────────────────────────
    port::Int          = 8000
    host::String       = "127.0.0.1"
    request_timeout_s::Int  = 120
    rate_limit_rpm::Int     = 30      # requests per minute per IP

    # ── Storage ────────────────────────────────────────────────────
    db_path::String         = "vector_db.jld2"
    docs_folder::String     = "data/documents"
    max_file_bytes::Int     = 50_000_000
end

end # module Config
