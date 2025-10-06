"""
Test if vector type from synthetic data generation causes allocations.

Hypothesis: The vectors created via division and replace!() have a different
concrete type that causes ForwardDiff allocations.
"""

using BenchmarkTools
using GLM
using DataFrames
using Tables
using FormulaCompiler
using FormulaCompiler: derivativeevaluator, derivative_modelrow!

include("../test/support/generate_large_synthetic_data.jl")

println("Testing vector type allocation differences...\n")

# Generate synthetic data
df_synth = generate_synthetic_dataset(1000; seed=08540)

# Create simple vectors the "normal" way
df_normal = DataFrame(
    response = rand([false, true], 1000),
    dists_p_inv = rand(1000),
    are_related_dists_a_inv = rand(1000)
)

println("=== Vector Type Comparison ===")
println("Synthetic dists_p_inv type: ", typeof(df_synth.dists_p_inv))
println("Normal dists_p_inv type: ", typeof(df_normal.dists_p_inv))
println("\nSynthetic eltype: ", eltype(df_synth.dists_p_inv))
println("Normal eltype: ", eltype(df_normal.dists_p_inv))
println()

# Test with both datasets
for (name, df) in [("Synthetic", df_synth), ("Normal", df_normal)]
    println("\n" * "="^80)
    println("Testing with $name data")
    println("="^80)

    # Fit simple model
    fx = @formula(response ~ dists_p_inv + are_related_dists_a_inv)
    model = fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())

    # Compile with copied data
    data_orig = Tables.columntable(df)
    data = NamedTuple{keys(data_orig)}(map(copy, values(data_orig)))

    compiled = compile_formula(model, data)

    # Test AD allocations
    vars = [:dists_p_inv, :are_related_dists_a_inv]
    de = derivativeevaluator(:ad, compiled, data, vars)
    J = Matrix{Float64}(undef, length(compiled), length(vars))

    # Warmup
    derivative_modelrow!(J, de, 1)

    # Benchmark
    b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000

    println("  Result: $(minimum(b).allocs) allocs, $(minimum(b).memory) bytes")
    println("  Time: $(minimum(b).time) ns")
end

println("\n\n=== Fresh Vector Test ===")
println("Creating completely fresh vectors from scratch...")

# Create dataset with absolutely fresh vectors
df_fresh = DataFrame(
    response = Vector{Bool}([rand(Bool) for _ in 1:1000]),
    dists_p_inv = Vector{Float64}([rand() for _ in 1:1000]),
    are_related_dists_a_inv = Vector{Float64}([rand() for _ in 1:1000])
)

println("Fresh vector type: ", typeof(df_fresh.dists_p_inv))

fx = @formula(response ~ dists_p_inv + are_related_dists_a_inv)
model = fit(GeneralizedLinearModel, fx, df_fresh, Bernoulli(), LogitLink())

data_orig = Tables.columntable(df_fresh)
data = NamedTuple{keys(data_orig)}(map(copy, values(data_orig)))

compiled = compile_formula(model, data)

vars = [:dists_p_inv, :are_related_dists_a_inv]
de = derivativeevaluator(:ad, compiled, data, vars)
J = Matrix{Float64}(undef, length(compiled), length(vars))

derivative_modelrow!(J, de, 1)
b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000

println("  Result: $(minimum(b).allocs) allocs, $(minimum(b).memory) bytes")
println("  Time: $(minimum(b).time) ns")
