# test/quick_test.jl
# Quick smoke test for development

using EfficientModelMatrices
using DataFrames, GLM, Tables, CategoricalArrays, Random

println("🚀 Quick Smoke Test")
println("=" ^ 30)

# Create minimal test data
Random.seed!(42)
df = DataFrame(
    x = [1.0, 2.0, 3.0, 4.0],
    y = [1.0, 4.0, 9.0, 16.0],
    group = categorical(["A", "B", "A", "B"])
)

data = Tables.columntable(df)

# Test basic functionality
println("Testing basic compilation...")
model = lm(@formula(y ~ x * group), df)
compiled = compile_formula(model)
row_vec = Vector{Float64}(undef, length(compiled))

# Test evaluation
println("Testing evaluation...")
compiled(row_vec, data, 1)
expected = modelmatrix(model)[1, :]

if isapprox(row_vec, expected, rtol=1e-12)
    println("✅ Basic functionality works")
else
    println("❌ Basic functionality failed")
    println("Expected: $expected")
    println("Got:      $row_vec")
end

# Test zero allocations
println("Testing zero allocations...")
allocs = @allocated compiled(row_vec, data, 1)
if allocs == 0
    println("✅ Zero allocations confirmed")
else
    println("⚠️  Non-zero allocations: $allocs bytes")
end

# Test evaluator tree access
println("Testing evaluator tree access...")
if has_evaluator_access(compiled)
    evaluator = extract_root_evaluator(compiled)
    node_count = count_evaluator_nodes(compiled)
    vars = get_variable_dependencies(compiled)
    println("✅ Evaluator tree access works")
    println("   Nodes: $node_count")
    println("   Variables: $vars")
else
    println("❌ Evaluator tree access failed")
end

# Test modelrow interfaces
println("Testing modelrow interfaces...")
try
    # Pre-compiled
    row1 = Vector{Float64}(undef, length(compiled))
    modelrow!(row1, compiled, data, 1)
    
    # Cached
    row2 = Vector{Float64}(undef, length(compiled))
    modelrow!(row2, model, data, 1)
    
    # Allocating
    row3 = modelrow(model, data, 1)
    
    if row1 == row2 == row3
        println("✅ All modelrow interfaces work")
    else
        println("❌ Modelrow interfaces inconsistent")
    end
catch e
    println("❌ Modelrow interfaces failed: $e")
end

# Test scenarios
println("Testing scenarios...")
try
    scenario = create_scenario("test", data; x = 10.0)
    modelrow!(row_vec, compiled, scenario, 1)
    println("✅ Scenarios work")
catch e
    println("❌ Scenarios failed: $e")
end

# Test performance
println("Testing performance...")
eval_time = @elapsed compiled(row_vec, data, 1)
if eval_time < 0.001
    println("✅ Performance good: $(round(eval_time*1e6, digits=1))μs")
else
    println("⚠️  Performance slow: $(round(eval_time*1e3, digits=1))ms")
end

println("\n🎉 Quick test complete!")
println("=" ^ 30)
