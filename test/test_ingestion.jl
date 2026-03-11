using Test
using StaticArrays

@testset "Ingestion Module" begin

    @testset "compress_to_binary" begin
        # All positive → all 1s → 0xFF per byte
        embedding = Float32[1.0 for _ in 1:16]
        result = Cerebro.compress_to_binary(embedding)
        @test length(result) == 2
        @test all(b -> b == 0xFF, result)

        # All negative → all 0s → 0x00 per byte
        embedding_neg = Float32[-1.0 for _ in 1:16]
        result_neg = Cerebro.compress_to_binary(embedding_neg)
        @test all(b -> b == 0x00, result_neg)

        # All zeros → all 0s (0 is NOT > 0)
        embedding_zero = Float32[0.0 for _ in 1:16]
        result_zero = Cerebro.compress_to_binary(embedding_zero)
        @test all(b -> b == 0x00, result_zero)

        # Mixed: first 8 positive, next 8 negative → [0xFF, 0x00]
        embedding_mixed = vcat(Float32[1.0 for _ in 1:8], Float32[-1.0 for _ in 1:8])
        result_mixed = Cerebro.compress_to_binary(embedding_mixed)
        @test result_mixed == UInt8[0xFF, 0x00]

        # Dimension validation
        @test_throws ArgumentError Cerebro.compress_to_binary(Float32[1.0, 2.0, 3.0])
    end

    @testset "compress_to_binary! (in-place)" begin
        embedding = Float32[1.0 for _ in 1:32]
        out = Vector{UInt8}(undef, 4)
        Cerebro.compress_to_binary!(out, embedding)
        @test all(b -> b == 0xFF, out)

        # Buffer too small
        small_out = Vector{UInt8}(undef, 1)
        @test_throws ArgumentError Cerebro.compress_to_binary!(small_out, embedding)
    end

    @testset "_chunk_text" begin
        text = "The quick brown fox jumps over the lazy dog and runs away"
        # 11 words total

        # Basic chunking
        chunks = Cerebro.Ingestion._chunk_text(text, 5, 0)
        @test length(chunks) >= 2
        @test all(c -> !isempty(c), chunks)

        # With overlap
        chunks_overlap = Cerebro.Ingestion._chunk_text(text, 5, 2)
        @test length(chunks_overlap) >= length(chunks)  # overlap produces more or equal chunks

        # Empty text
        chunks_empty = Cerebro.Ingestion._chunk_text("", 5, 0)
        @test isempty(chunks_empty)

        # Single word
        chunks_single = Cerebro.Ingestion._chunk_text("Hello", 5, 0)
        @test length(chunks_single) == 1
        @test chunks_single[1] == "Hello"
    end

    @testset "_chunk_text_structured" begin
        text = """First paragraph with multiple words here.

Second paragraph is here too.

Third paragraph finishes the document."""

        chunks = Cerebro.Ingestion._chunk_text_structured(text, 20, 0)
        @test length(chunks) >= 1
        @test all(c -> !isempty(c), chunks)

        # Empty text
        chunks_empty = Cerebro.Ingestion._chunk_text_structured("", 5, 0)
        @test isempty(chunks_empty)
    end

    @testset "Document and VectorDB structures" begin
        # Test VectorDB auto-builds parent_lookup
        p1 = Cerebro.ParentChunk("p1", "doc1", "text1")
        p2 = Cerebro.ParentChunk("p2", "doc1", "text2")
        c1 = Cerebro.ChildChunk("c1", "p1", "child_text1")
        index = Cerebro.BinaryIndex{2}(SVector{2,UInt8}[SVector{2,UInt8}(0x00, 0x00)])

        db = Cerebro.VectorDB([p1, p2], [c1], index)
        @test haskey(db.parent_lookup, "p1")
        @test haskey(db.parent_lookup, "p2")
        @test db.parent_lookup["p1"].text == "text1"
        @test length(db.parent_lookup) == 2
    end

    @testset "load_documents" begin
        # Non-existent directory
        docs = Cerebro.load_documents("/tmp/cerebro_test_nonexistent_dir_12345")
        @test isempty(docs)

        # Real directory with a test file
        test_dir = mktempdir()
        write(joinpath(test_dir, "test.md"), "Hello world")
        write(joinpath(test_dir, "test.txt"), "Another file")
        write(joinpath(test_dir, "test.pdf"), "Should be ignored")

        docs = Cerebro.load_documents(test_dir)
        @test length(docs) == 2
        @test any(d -> contains(d.content, "Hello world"), docs)

        # File size limit
        big_file = joinpath(test_dir, "big.md")
        write(big_file, repeat("x", 1000))
        docs_limited = Cerebro.load_documents(test_dir; max_file_bytes=500)
        @test length(docs_limited) == 2  # big file should be skipped

        rm(test_dir; recursive=true)
    end

    @testset "hierarchical_chunk" begin
        docs = [Cerebro.Document("test", repeat("word ", 500))]
        parents, children = Cerebro.hierarchical_chunk(docs; parent_words=100, child_words=25)
        @test length(parents) > 0
        @test length(children) > 0
        # Each parent should have at least one child
        parent_ids = Set(p.id for p in parents)
        child_parent_ids = Set(c.parent_id for c in children)
        @test child_parent_ids ⊆ parent_ids
    end
end
