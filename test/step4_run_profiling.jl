# run_profiling.jl
# Script to run allocation profiling on Step 4

using Revise
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, Random
using BenchmarkTools, Profile

using FormulaCompiler:
    compile_formula_specialized_complete, execute_interaction_operation!,
    evaluate_unified_component!, execute_to_scratch!,
    execute_complete_constant_operations!,
    execute_complete_continuous_operations!,
    execute_categorical_operations!,
    execute_interaction_operations!,
    execute_linear_function_operations!

# Include the profiling tools
include("step4_profiling.jl")

# Set seed for reproducible results
Random.seed!(12345)

println("🔍 Starting Step 4 Allocation Profiling...")
println()

# Test 1: Run comprehensive profiling on all critical cases
println("1. COMPREHENSIVE PROFILING")
println("="^50)
allocation_results = run_comprehensive_allocation_profiling()

println("\n2. BUFFER ALLOCATION PATTERN TESTING")
println("="^50)
buffer_results = test_buffer_allocation_patterns()

println("\n3. DETAILED ANALYSIS OF WORST CASES")
println("="^50)

# Create test data for detailed analysis
n = 200
df = DataFrame(
    x = randn(n),
    y = randn(n), 
    z = abs.(randn(n)) .+ 0.01,
    w = randn(n),
    group3 = categorical(rand(["A", "B", "C"], n)),           
    group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      
    binary = categorical(rand(["Yes", "No"], n)),
    response = randn(n)
)
data = Tables.columntable(df)

# Focus on the cases that still allocate significantly
worst_cases = [
    @formula(response ~ log(z) * group4),                    # Test 6: Still 304 bytes
    @formula(response ~ x * y * group3 + log(z) * group4),   # Test 9: Your formula, 400 bytes
    @formula(response ~ x * y * z * w),                      # Test 12: 544 bytes
]

for formula in worst_cases
    println("\nDETAILED ANALYSIS: $formula")
    println("-" ^ 60)
    
    # Compile and get detailed breakdown
    model = fit(LinearModel, formula, df)
    specialized = compile_formula_specialized_complete(model, data)
    output = Vector{Float64}(undef, length(specialized))
    
    # Test if we can isolate interaction component issues
    if !isempty(specialized.data.interactions)
        println("This formula has $(length(specialized.data.interactions)) interactions")
        for (i, interaction) in enumerate(specialized.data.interactions)
            println("  Interaction $i: $(length(interaction.components)) components, $(length(interaction.kronecker_pattern)) terms")
            
            # Test individual interaction execution
            int_allocs = @allocated begin
                for j in 1:100
                    execute_interaction_operation!(interaction, specialized.data.interaction_scratch, output, data, 1)
                end
            end
            println("    Allocations: $(int_allocs/100) bytes per call")
            
            # Profile individual components if this interaction allocates
            if int_allocs > 0
                profile_interaction_components_individually(interaction, specialized.data.interaction_scratch, output, data, 1)
            end
        end
    end
end

println("\n" * "="^80)
println("🎯 PROFILING COMPLETE - CHECK RESULTS ABOVE")
println("="^80)

println("\n📊 SUMMARY:")
println("- Look for the highest allocation sources in each test")
println("- Check if allocations come from functions vs interactions vs buffers")
println("- Identify the specific allocation patterns to fix in Phase 2")
println()
println("💡 Next steps based on results:")
println("- If functions allocate: Fix function buffer management")  
println("- If interactions allocate: Fix interaction scratch management")
println("- If buffers allocate: Fix pre-allocation strategy")
println("- If views allocate: Use direct indexing instead")