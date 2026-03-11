module App

using Genie
using Genie.Router
using Genie.Renderer.Html
using Genie.Renderer.Json
using Genie.Requests

using ..Ingestion
using ..Backend

export start_server, AppConfig

const DB_PATH = "vector_db.bin"

mutable struct AppConfig
    db::Any
    embedding_model::String
    chat_model::String
end

function init_db(folder_path::String, embedding_model::String)
    if isfile(DB_PATH)
        @info "Loading existing database from $DB_PATH..."
        db = load_database(DB_PATH)
        @info "Database loaded: $(length(db.parents)) parents, $(length(db.children)) children."
        return db
    else
        @info "No database found. Building from $folder_path..."
        mkpath(folder_path)
        docs = load_documents(folder_path)
        if isempty(docs)
            @info "No documents found in $folder_path. Please add some .txt or .md files to test."
            return VectorDB{0}(ParentChunk[], ChildChunk[], BinaryIndex{0}(SVector{0,Int8}[]))
        end
        @info "Chunking $(length(docs)) documents..."
        parents, children = hierarchical_chunk(docs)
        @info "Embedding and quantizing $(length(children)) child chunks..."
        index = embed_and_quantize(children; model=embedding_model)
        @info "Saving database..."
        db = VectorDB(parents, children, index)
        save_database(DB_PATH, db)
        @info "Database loaded: $(length(db.parents)) parents, $(length(db.children)) children."
        return db
    end
end

function setup_routes(config::AppConfig)
    route("/") do
        html(read(joinpath(dirname(@__DIR__), "public", "index.html"), String))
    end

    route("/api/chat", method=POST) do
        req_data = jsonpayload()
        if !haskey(req_data, "query")
            return json(Dict("error" => "query is required"))
        end

        try
            query = req_data["query"]
            db = config.db

            if isempty(db.parents) || length(db.index.vectors) == 0
                return json(Dict("answer" => "The database is empty. Please add documents to the folder and restart.", "sources" => []))
            end

            @info "Retrieving context for query: $query"
            top_parents = retrieve_context(query, db; k=3, model=config.embedding_model)

            @info "Generating answer..."
            answer = generate_answer(query, top_parents; model=config.chat_model)

            sources = [Dict("id" => p.id, "doc_id" => p.doc_id) for p in top_parents]
            @info "Sending response"

            return json(Dict("answer" => answer, "sources" => sources))
        catch e
            @error "Failed to generate answer" exception=(e, catch_backtrace())
            return json(Dict("error" => "An internal server error occurred while processing your request."))
        end
    end
end

function start_server(; port=8000, host="127.0.0.1", folder_path="data/documents", embedding_model="embeddinggemma:300m-qat-q4_0", chat_model="gemma3:1b")
    @info "=== Starting Cerebro ==="
    db = init_db(folder_path, embedding_model)
    config = AppConfig(db, embedding_model, chat_model)
    setup_routes(config)

    # Configure Genie
    Genie.config.run_as_server = true
    Genie.config.server_port = port
    Genie.config.server_host = host

    @info "Server running at http://$host:$port"
    Genie.up()
end

end # module
