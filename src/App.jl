module App

using HTTP
using JSON

using StaticArrays
using ..Config: CerebroConfig
using ..Ingestion
using ..Backend

export start_server

# ─── Rate Limiting (Token Bucket per IP) ──────────────────────────

mutable struct TokenBucket
    tokens::Float64
    last_refill::Float64
    const max_tokens::Float64
    const refill_rate::Float64   # tokens per second
end

struct RateLimiter
    buckets::Dict{String, TokenBucket}
    lock::ReentrantLock
    max_rpm::Int
end

function RateLimiter(max_rpm::Int)
    RateLimiter(Dict{String, TokenBucket}(), ReentrantLock(), max_rpm)
end

function check_rate_limit!(limiter::RateLimiter, ip::String)::Bool
    lock(limiter.lock) do
        now = time()
        rate = limiter.max_rpm / 60.0

        if !haskey(limiter.buckets, ip)
            limiter.buckets[ip] = TokenBucket(limiter.max_rpm - 1.0, now, Float64(limiter.max_rpm), rate)
            return true
        end

        bucket = limiter.buckets[ip]
        elapsed = now - bucket.last_refill
        bucket.tokens = min(bucket.max_tokens, bucket.tokens + elapsed * bucket.refill_rate)
        bucket.last_refill = now

        if bucket.tokens >= 1.0
            bucket.tokens -= 1.0
            return true
        end
        return false
    end
end

# ─── Type-Stable App State ────────────────────────────────────────

mutable struct AppState{N}
    db::VectorDB{N}
    const config::CerebroConfig
    const lock::ReentrantLock
    const rate_limiter::RateLimiter
    const index_html::String    # cached HTML content
end

# ─── Ollama Health Check ──────────────────────────────────────────

function check_ollama(; timeout::Int=5)
    try
        response = HTTP.get("http://localhost:11434/api/tags"; connect_timeout=timeout, readtimeout=timeout)
        return response.status == 200
    catch
        return false
    end
end

# ─── Database Initialization ─────────────────────────────────────

function init_db(config::CerebroConfig)
    db_path = config.db_path
    folder_path = config.docs_folder

    if isfile(db_path)
        @info "Loading existing database from $db_path..."
        db = load_database(db_path)
        @info "Database loaded" parents=length(db.parents) children=length(db.children)
        return db
    end

    @info "No database found. Building from $folder_path..."
    mkpath(folder_path)
    docs = load_documents(folder_path; max_file_bytes=config.max_file_bytes)

    if isempty(docs)
        @warn "No documents found in $folder_path. Add .txt or .md files and restart."
        return VectorDB(ParentChunk[], ChildChunk[], BinaryIndex{0}(SVector{0,UInt8}[]))
    end

    @info "Chunking $(length(docs)) documents..."
    parents, children = hierarchical_chunk(docs;
        parent_words=config.parent_words,
        child_words=config.child_words,
        parent_overlap=config.parent_overlap,
        child_overlap=config.child_overlap
    )

    @info "Embedding and quantizing $(length(children)) child chunks..."
    index = embed_and_quantize(children; config=config)

    db = VectorDB(parents, children, index)
    @info "Saving database..."
    save_database(db_path, db)
    @info "Database ready" parents=length(db.parents) children=length(db.children)
    return db
end

# ─── HTTP Helpers ─────────────────────────────────────────────────

json_response(data; status=200) = HTTP.Response(status,
    ["Content-Type" => "application/json", "Access-Control-Allow-Origin" => "*"],
    body=JSON.json(data))

html_response(content) = HTTP.Response(200,
    ["Content-Type" => "text/html; charset=utf-8"],
    body=content)

function get_client_ip(request::HTTP.Request)
    for (name, value) in request.headers
        lowercase(name) == "x-forwarded-for" && return first(split(value, ","))
        lowercase(name) == "x-real-ip" && return value
    end
    return "unknown"
end

function parse_json_body(request::HTTP.Request)
    try
        return JSON.parse(String(request.body))
    catch
        return nothing
    end
end

# ─── Request Router ───────────────────────────────────────────────

function handle_request(state::AppState, request::HTTP.Request)
    method = request.method
    path = HTTP.URI(request.target).path

    # CORS preflight
    if method == "OPTIONS"
        return HTTP.Response(204, [
            "Access-Control-Allow-Origin" => "*",
            "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers" => "Content-Type",
        ])
    end

    # Route dispatch
    if method == "GET" && path == "/"
        return html_response(state.index_html)
    elseif method == "GET" && path == "/health"
        return handle_health(state)
    elseif method == "POST" && path == "/api/chat"
        return handle_chat(state, request)
    else
        return json_response(Dict("error" => "Not found"); status=404)
    end
end

# ─── Route Handlers ───────────────────────────────────────────────

function handle_health(state::AppState)
    ollama_ok = check_ollama()
    db_ok = length(state.db.index.vectors) > 0

    return json_response(Dict(
        "status" => (ollama_ok && db_ok) ? "healthy" : "degraded",
        "ollama" => ollama_ok ? "connected" : "unreachable",
        "database" => Dict(
            "loaded" => db_ok,
            "parents" => length(state.db.parents),
            "children" => length(state.db.children)
        ),
        "version" => "0.2.0"
    ))
end

function handle_chat(state::AppState, request::HTTP.Request)
    # Rate limiting
    ip = get_client_ip(request)
    if !check_rate_limit!(state.rate_limiter, ip)
        return json_response(Dict("error" => "Rate limit exceeded. Please wait before sending another request."); status=429)
    end

    # Parse body
    body = parse_json_body(request)
    if isnothing(body) || !haskey(body, "query")
        return json_response(Dict("error" => "query is required"); status=400)
    end

    try
        query = strip(String(body["query"]))

        # Input validation
        if isempty(query)
            return json_response(Dict("error" => "query cannot be empty"); status=400)
        end
        if length(query) > 2000
            return json_response(Dict("error" => "query too long (max 2000 characters)"); status=400)
        end

        db = lock(state.lock) do
            state.db
        end

        if isempty(db.parents) || length(db.index.vectors) == 0
            return json_response(Dict(
                "answer" => "The database is empty. Please add documents to the folder and restart.",
                "sources" => []
            ))
        end

        # Process the query
        @info "Retrieving context" query=first(query, 80)
        top_parents = retrieve_context(query, db; config=state.config)

        @info "Generating answer..."
        answer = generate_answer(query, top_parents; config=state.config)

        sources = [Dict("id" => p.id, "doc_id" => p.doc_id) for p in top_parents]
        @info "Response sent"

        return json_response(Dict("answer" => answer, "sources" => sources))

    catch e
        @error "Failed to process request" exception=(e, catch_backtrace())
        return json_response(Dict("error" => "An internal server error occurred."); status=500)
    end
end

# ─── Server Entry Point ──────────────────────────────────────────

function start_server(; config::CerebroConfig=CerebroConfig())
    @info "=== Starting Cerebro v0.2.0 ==="
    @info "Configuration" embedding_model=config.embedding_model chat_model=config.chat_model port=config.port

    # Validate Ollama connectivity
    if !check_ollama()
        @warn "⚠️  Ollama is not reachable at localhost:11434. Queries will fail until Ollama is available."
    else
        @info "✅ Ollama is reachable"
    end

    # Initialize database
    db = init_db(config)
    N = length(db.index.vectors) > 0 ? length(db.index.vectors[1]) : 0

    # Cache HTML at startup
    html_path = joinpath(dirname(@__DIR__), "public", "index.html")
    index_html = isfile(html_path) ? read(html_path, String) : "<h1>Cerebro — index.html not found</h1>"

    state = AppState{N}(
        db,
        config,
        ReentrantLock(),
        RateLimiter(config.rate_limit_rpm),
        index_html
    )

    # Register shutdown handler
    atexit() do
        @info "Cerebro shutting down gracefully..."
    end

    addr = config.host
    port = config.port
    @info "Server running at http://$addr:$port"
    @info "Health check at http://$addr:$port/health"

    # Start HTTP server
    HTTP.serve(addr, port) do request::HTTP.Request
        handle_request(state, request)
    end
end

end # module App
