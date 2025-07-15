###############################################################################
# INTEGRATION EXAMPLE
###############################################################################

"""
Example showing the progression from convenient to high-performance usage.
"""
function performance_example()
    # Setup
    # using GLM, DataFrames, Tables, BenchmarkTools
    
    df = DataFrame(
        x = randn(1000), 
        y = randn(1000), 
        group = categorical(rand(["A", "B", "C"], 1000))
    )
    model = lm(@formula(y ~ x^2 * group), df)
    data = Tables.columntable(df)
    
    println("=== Performance Progression ===")
    
    # Level 1: Standard StatsModels (slow)
    println("1. Standard approach:")
    mm = modelmatrix(model)
    @btime $mm[1, :]  # ~1-10 μs, allocates
    
    # Level 2: Convenient modelrow! (fast, auto-caching)
    println("2. Convenient modelrow!:")
    row_vec = Vector{Float64}(undef, size(mm, 2))
    @btime modelrow!($row_vec, $model, $data, 1)  # ~100-500ns after first call
    
    # Level 3: Pre-compiled (fastest)
    println("3. Pre-compiled:")
    compiled = compile_formula(model)
    @btime $compiled($row_vec, $data, 1)  # ~50-100ns, zero allocations
    
    # Level 4: Batch processing
    println("4. Batch processing:")
    matrix = Matrix{Float64}(undef, 100, length(compiled))
    @btime modelrow!($matrix, $model, $data, 1:100)
    
    return compiled
end

###############################################################################
# TESTING UTILITIES
###############################################################################

"""
    test_modelrow_interface()

Test the complete modelrow! interface for correctness and performance.
"""
function test_modelrow_interface()
    using Test
    
    # Setup test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0],
        y = [1.0, 4.0, 9.0],
        group = categorical(["A", "B", "A"])
    )
    model = lm(@formula(y ~ x * group), df)
    data = Tables.columntable(df)
    mm = modelmatrix(model)
    
    @testset "modelrow! Interface" begin
        
        @testset "Single Row Evaluation" begin
            row_vec = Vector{Float64}(undef, size(mm, 2))
            
            # Test row 1
            result = modelrow!(row_vec, model, data, 1)
            @test result === row_vec  # Returns same vector
            @test isapprox(row_vec, mm[1, :], atol=1e-12)
            
            # Test row 2  
            modelrow!(row_vec, model, data, 2)
            @test isapprox(row_vec, mm[2, :], atol=1e-12)
        end
        
        @testset "Batch Evaluation" begin
            matrix = Matrix{Float64}(undef, 3, size(mm, 2))
            
            result = modelrow!(matrix, model, data, 1:3)
            @test result === matrix  # Returns same matrix
            @test isapprox(matrix, mm, atol=1e-12)
        end
        
        @testset "Allocating Versions" begin
            # Single row
            row_vec = modelrow(model, data, 1)
            @test isapprox(row_vec, mm[1, :], atol=1e-12)
            
            # Multiple rows
            matrix = modelrow(model, data, 1:3)
            @test isapprox(matrix, mm, atol=1e-12)
        end
        
        @testset "Caching Behavior" begin
            # Clear cache
            clear_model_cache!()
            
            # First call should compile
            row_vec = Vector{Float64}(undef, size(mm, 2))
            @test_nowarn modelrow!(row_vec, model, data, 1)
            
            # Second call should use cache (test that it doesn't error)
            @test_nowarn modelrow!(row_vec, model, data, 2)
        end
        
        @testset "Performance Characteristics" begin
            row_vec = Vector{Float64}(undef, size(mm, 2))
            
            # Warmup
            for i in 1:10
                modelrow!(row_vec, model, data, 1)
            end
            
            # Test allocations
            allocs = @allocated modelrow!(row_vec, model, data, 1)
            @test allocs < 100  # Should be very low after compilation
            
            println("   Allocations: $allocs bytes")
        end
    end
    
    return true
end
