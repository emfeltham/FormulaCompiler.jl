# Test if synthetic data causes AD allocations

using FormulaCompiler, GLM, DataFrames, Tables, BenchmarkTools
include("../test/support/generate_large_synthetic_data.jl")

println("Comparing synthetic vs normal data...")
println("=" ^ 60)

# Test 1: Synthetic data (from generate_large_synthetic_data.jl)
println("\nTest 1: Synthetic data")
df_synth = generate_synthetic_dataset(1000; seed=08540)
model_synth = glm(@formula(response ~ socio4 + dists_p_inv + are_related_dists_a_inv), df_synth, Bernoulli(), LogitLink())
data_synth = Tables.columntable(df_synth)
compiled_synth = compile_formula(model_synth, data_synth)
de_ad_synth = derivativeevaluator(:ad, compiled_synth, data_synth, [:dists_p_inv, :are_related_dists_a_inv])
J_synth = Matrix{Float64}(undef, length(compiled_synth), 2)
derivative_modelrow!(J_synth, de_ad_synth, 1)
b_synth = @benchmark derivative_modelrow!($J_synth, $de_ad_synth, 1) samples=400
println("  Synthetic data: $(minimum(b_synth).allocs) allocs, $(minimum(b_synth).memory) bytes")

# Test 2: Normal data with same variable names
println("\nTest 2: Normal data (same formula, simple generation)")
df_normal = DataFrame(
    response = rand([0, 1], 1000),
    socio4 = rand([true, false], 1000),
    dists_p_inv = randn(1000),
    are_related_dists_a_inv = randn(1000)
)
model_normal = glm(@formula(response ~ socio4 + dists_p_inv + are_related_dists_a_inv), df_normal, Bernoulli(), LogitLink())
data_normal = Tables.columntable(df_normal)
compiled_normal = compile_formula(model_normal, data_normal)
de_ad_normal = derivativeevaluator(:ad, compiled_normal, data_normal, [:dists_p_inv, :are_related_dists_a_inv])
J_normal = Matrix{Float64}(undef, length(compiled_normal), 2)
derivative_modelrow!(J_normal, de_ad_normal, 1)
b_normal = @benchmark derivative_modelrow!($J_normal, $de_ad_normal, 1) samples=400
println("  Normal data: $(minimum(b_normal).allocs) allocs, $(minimum(b_normal).memory) bytes")

# Test 3: Check data types
println("\nTest 3: Data type comparison")
println("  Synthetic socio4 type: $(typeof(data_synth.socio4))")
println("  Normal socio4 type: $(typeof(data_normal.socio4))")
println("  Synthetic dists_p_inv type: $(typeof(data_synth.dists_p_inv))")
println("  Normal dists_p_inv type: $(typeof(data_normal.dists_p_inv))")
