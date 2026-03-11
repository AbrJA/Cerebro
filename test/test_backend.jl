using Test
using StaticArrays

@testset "Backend Module" begin

    @testset "hamming_distance" begin
        # Same values → distance 0
        @test Cerebro.Backend.hamming_distance(0x00, 0x00) == 0
        @test Cerebro.Backend.hamming_distance(0xFF, 0xFF) == 0

        # All bits different → distance 8 (for UInt8)
        @test Cerebro.Backend.hamming_distance(0x00, 0xFF) == 8

        # Single bit different
        @test Cerebro.Backend.hamming_distance(0x00, 0x01) == 1
        @test Cerebro.Backend.hamming_distance(0xFF, 0xFE) == 1

        # Symmetry
        @test Cerebro.Backend.hamming_distance(0x0F, 0xF0) == Cerebro.Backend.hamming_distance(0xF0, 0x0F)

        # SVector hamming distance
        v1 = SVector{4, UInt8}(0xFF, 0x00, 0xFF, 0x00)
        v2 = SVector{4, UInt8}(0x00, 0xFF, 0x00, 0xFF)
        @test Cerebro.Backend.hamming_distance(v1, v2) == 32  # 4 bytes * 8 bits each
        
        # Same SVector → 0
        @test Cerebro.Backend.hamming_distance(v1, v1) == 0
    end

    @testset "MaxHeap" begin
        # Basic: insert k elements, heap should contain all of them
        heap = Cerebro.Backend.MaxHeap(3)
        Cerebro.Backend.heap_push!(heap, 5 => 1)
        Cerebro.Backend.heap_push!(heap, 3 => 2)
        Cerebro.Backend.heap_push!(heap, 7 => 3)
        
        distances = sort([p.first for p in heap.data])
        @test distances == [3, 5, 7]

        # Insert a smaller value → should replace the max
        Cerebro.Backend.heap_push!(heap, 1 => 4)
        distances_after = sort([p.first for p in heap.data])
        @test 7 ∉ distances_after  # 7 should be gone
        @test 1 ∈ distances_after  # 1 should be in

        # Insert a larger value → should be ignored
        Cerebro.Backend.heap_push!(heap, 100 => 5)
        @test 100 ∉ [p.first for p in heap.data]
    end

    @testset "MaxHeap k=1" begin
        heap = Cerebro.Backend.MaxHeap(1)
        Cerebro.Backend.heap_push!(heap, 10 => 1)
        Cerebro.Backend.heap_push!(heap, 5 => 2)
        Cerebro.Backend.heap_push!(heap, 3 => 3)
        Cerebro.Backend.heap_push!(heap, 20 => 4)
        @test heap.data[1] == (3 => 3)
    end

    @testset "_k_closest" begin
        # Create synthetic database
        db = [
            SVector{2, UInt8}(0x00, 0x00),  # distance 0 from query
            SVector{2, UInt8}(0xFF, 0xFF),  # distance 16 from query
            SVector{2, UInt8}(0x01, 0x00),  # distance 1 from query
            SVector{2, UInt8}(0x03, 0x00),  # distance 2 from query
            SVector{2, UInt8}(0x0F, 0x0F),  # distance 8 from query
        ]
        query = SVector{2, UInt8}(0x00, 0x00)

        result = Cerebro.Backend._k_closest(db, query, 3)
        distances = sort([p.first for p in result])
        @test distances == [0, 1, 2]  # top 3 closest

        # k = 1 → should return the closest
        result_1 = Cerebro.Backend._k_closest(db, query, 1)
        @test result_1[1].first == 0
        @test result_1[1].second == 1  # index of [0x00, 0x00]
    end

    @testset "_k_closest with startind" begin
        db = [
            SVector{2, UInt8}(0x00, 0x00),
            SVector{2, UInt8}(0xFF, 0xFF),
        ]
        query = SVector{2, UInt8}(0x00, 0x00)

        result = Cerebro.Backend._k_closest(db, query, 1; startind=100)
        @test result[1].second == 100  # should use the offset
    end

    @testset "k_closest_parallel" begin
        # Create a larger synthetic database
        db = [SVector{4, UInt8}(rand(UInt8, 4)) for _ in 1:200]
        query = db[42]  # use one of them as query

        result = Cerebro.Backend.k_closest_parallel(db, query, 5; min_parallel_threshold=50)
        # The closest match should be the query itself (distance 0)
        @test result[1].first == 0
        @test result[1].second == 42

        # Results should be sorted by distance
        for i in 2:length(result)
            @test result[i].first >= result[i-1].first
        end
    end

    @testset "k_closest_parallel small DB (skips parallelism)" begin
        db = [SVector{2, UInt8}(rand(UInt8, 2)) for _ in 1:10]
        query = db[3]

        # With high threshold, should skip parallelism
        result = Cerebro.Backend.k_closest_parallel(db, query, 3; min_parallel_threshold=10000)
        @test result[1].first == 0
        @test result[1].second == 3
    end
end
