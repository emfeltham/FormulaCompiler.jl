###############################################################################
# PERFORMANCE COMPARISON
###############################################################################

"""
Demonstrate the performance trade-offs between different approaches.
"""
function performance_comparison()
    using BenchmarkTools, DataFrames, GLM, Tables
    
    # Setup
    df = DataFrame(
        x = randn(1000),
        y = randn(1000), 
        group = categorical(rand(["A", "B", "C"], 1000))
    )
    model = lm(@formula(y ~ x^2 * group), df)
    data = Tables.columntable(df)
    row_vec = Vector{Float64}(undef, size(modelmatrix(model), 2))
    
    println("=== Performance Comparison ===")
    
    # 1. Standard StatsModels
    println("1. Standard StatsModels.modelmatrix:")
    mm = modelmatrix(model)
    @btime $mm[1, :]
    
    # 2. Convenient modelrow! (with caching)
    println("2. Convenient modelrow! (cached):")
    @btime modelrow!($row_vec, $model, $data, 1; cache=true)
    
    # 3. Pre-compiled version (zero-allocation target)
    println("3. Pre-compiled (zero-allocation):")
    compiled = compile_formula(model)
    @btime modelrow!($row_vec, $compiled, $data, 1)
    
    # 4. Object-based evaluator (ultimate performance)
    println("4. ModelRowEvaluator (ultimate):")
    evaluator = ModelRowEvaluator(model, df)
    @btime $evaluator(1)
    
    # 5. Check allocations specifically
    println("\n=== Allocation Check ===")
    
    allocs_convenient = @allocated modelrow!(row_vec, model, data, 1; cache=true)
    allocs_precompiled = @allocated modelrow!(row_vec, compiled, data, 1)
    allocs_evaluator = @allocated evaluator(1)
    
    println("Convenient:   $allocs_convenient bytes")
    println("Pre-compiled: $allocs_precompiled bytes") 
    println("Evaluator:    $allocs_evaluator bytes")
    
    return (allocs_convenient, allocs_precompiled, allocs_evaluator)
end
