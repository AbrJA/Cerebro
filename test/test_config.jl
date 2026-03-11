using Test

@testset "CerebroConfig" begin
    # Test defaults
    config = Cerebro.CerebroConfig()
    @test config.embedding_model == "embeddinggemma:300m-qat-q4_0"
    @test config.chat_model == "gemma3:1b"
    @test config.port == 8000
    @test config.host == "127.0.0.1"
    @test config.parent_words == 300
    @test config.child_words == 75
    @test config.retrieval_k == 3
    @test config.rate_limit_rpm == 30
    @test config.db_path == "vector_db.jld2"

    # Test custom config
    config2 = Cerebro.CerebroConfig(port=9000, retrieval_k=5, chat_model="llama3")
    @test config2.port == 9000
    @test config2.retrieval_k == 5
    @test config2.chat_model == "llama3"
    @test config2.embedding_model == "embeddinggemma:300m-qat-q4_0"  # still default
end
