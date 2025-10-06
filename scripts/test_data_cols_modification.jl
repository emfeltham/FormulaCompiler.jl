# Test different modifications to data_cols to find what causes AD allocations

using FormulaCompiler, GLM, DataFrames, Tables, BenchmarkTools
include("../test/support/generate_large_synthetic_data.jl")

println("Testing data_cols modifications...")
println("=" ^ 60)

# Generate synthetic data
df_synth = generate_synthetic_dataset(1000; seed=08540)
data_synth = Tables.columntable(df_synth)

# Fit model once
model = glm(@formula(response ~ socio4 + dists_p_inv + are_related_dists_a_inv), df_synth, Bernoulli(), LogitLink())

# Helper function to test a data_cols variant
function test_variant(name, data_cols)
    compiled = compile_formula(model, data_cols)
    de_ad = derivativeevaluator(:ad, compiled, data_cols, [:dists_p_inv, :are_related_dists_a_inv])
    J = Matrix{Float64}(undef, length(compiled), 2)

    # Warmup
    derivative_modelrow!(J, de_ad, 1)

    # Benchmark
    b = @benchmark derivative_modelrow!($J, $de_ad, 1) samples=400
    println("$name: $(minimum(b).allocs) allocs, $(minimum(b).memory) bytes")
    return minimum(b).allocs
end

# Test 1: Original synthetic data
println("\nTest 1: Original synthetic data_cols")
test_variant("  Original", data_synth)

# Test 2: Copy all columns
println("\nTest 2: Copy all columns to break any views/references")
data_copied = (
    response = copy(data_synth.response),
    socio4 = copy(data_synth.socio4),
    dists_p_inv = copy(data_synth.dists_p_inv),
    are_related_dists_a_inv = copy(data_synth.are_related_dists_a_inv),
)
test_variant("  Copied", data_copied)

# Test 3: Keep only needed columns (subset of original)
println("\nTest 3: Minimal columns (subset of original vectors)")
data_minimal = (
    response = data_synth.response,
    socio4 = data_synth.socio4,
    dists_p_inv = data_synth.dists_p_inv,
    are_related_dists_a_inv = data_synth.are_related_dists_a_inv,
)
test_variant("  Minimal", data_minimal)

# Test 4: Collect vectors (convert from SubArray if any)
println("\nTest 4: Collect vectors to ensure concrete arrays")
data_collected = (
    response = collect(data_synth.response),
    socio4 = collect(data_synth.socio4),
    dists_p_inv = collect(data_synth.dists_p_inv),
    are_related_dists_a_inv = collect(data_synth.are_related_dists_a_inv),
)
test_variant("  Collected", data_collected)

# Test 5: Fresh vectors with same values
println("\nTest 5: Fresh vectors (comprehension)")
data_fresh = (
    response = [v for v in data_synth.response],
    socio4 = [v for v in data_synth.socio4],
    dists_p_inv = [v for v in data_synth.dists_p_inv],
    are_related_dists_a_inv = [v for v in data_synth.are_related_dists_a_inv],
)
test_variant("  Fresh", data_fresh)

# Test 6: Check actual types
println("\nType investigation:")
println("  response: $(typeof(data_synth.response))")
println("  socio4: $(typeof(data_synth.socio4))")
println("  dists_p_inv: $(typeof(data_synth.dists_p_inv))")
println("  are_related_dists_a_inv: $(typeof(data_synth.are_related_dists_a_inv))")
