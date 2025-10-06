"""
Investigate what's special about synthetic data values that causes allocations.
"""

using BenchmarkTools
using GLM
using DataFrames
using Tables
using FormulaCompiler
using FormulaCompiler: derivativeevaluator, derivative_modelrow!
using Statistics

include("../test/support/generate_large_synthetic_data.jl")

println("Investigating synthetic data values...\n")

# Generate synthetic data
df_synth = generate_synthetic_dataset(1000; seed=08540)

println("=== Synthetic Data Statistics ===")
println("dists_p_inv:")
println("  min: ", minimum(df_synth.dists_p_inv))
println("  max: ", maximum(df_synth.dists_p_inv))
println("  mean: ", mean(df_synth.dists_p_inv))
println("  any NaN: ", any(isnan, df_synth.dists_p_inv))
println("  any Inf: ", any(isinf, df_synth.dists_p_inv))
println("  any subnormal: ", any(x -> !iszero(x) && abs(x) < floatmin(Float64), df_synth.dists_p_inv))
println()

println("are_related_dists_a_inv:")
println("  min: ", minimum(df_synth.are_related_dists_a_inv))
println("  max: ", maximum(df_synth.are_related_dists_a_inv))
println("  mean: ", mean(df_synth.are_related_dists_a_inv))
println("  any NaN: ", any(isnan, df_synth.are_related_dists_a_inv))
println("  any Inf: ", any(isinf, df_synth.are_related_dists_a_inv))
println("  any subnormal: ", any(x -> !iszero(x) && abs(x) < floatmin(Float64), df_synth.are_related_dists_a_inv))
println()

# Test: Use exact same values but in fresh vectors
println("\n=== Test 1: Copy exact synthetic values to fresh vectors ===")
df_copy_values = DataFrame(
    response = df_synth.response,
    dists_p_inv = Vector{Float64}(df_synth.dists_p_inv),
    are_related_dists_a_inv = Vector{Float64}(df_synth.are_related_dists_a_inv)
)

fx = @formula(response ~ dists_p_inv + are_related_dists_a_inv)
model = fit(GeneralizedLinearModel, fx, df_copy_values, Bernoulli(), LogitLink())

data_orig = Tables.columntable(df_copy_values)
data = NamedTuple{keys(data_orig)}(map(copy, values(data_orig)))

compiled = compile_formula(model, data)
vars = [:dists_p_inv, :are_related_dists_a_inv]
de = derivativeevaluator(:ad, compiled, data, vars)
J = Matrix{Float64}(undef, length(compiled), length(vars))

derivative_modelrow!(J, de, 1)
b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000
println("  Result: $(minimum(b).allocs) allocs, $(minimum(b).memory) bytes")

# Test: Use collect() on the columns
println("\n=== Test 2: Use collect() on synthetic columns ===")
df_collect = DataFrame(
    response = collect(df_synth.response),
    dists_p_inv = collect(df_synth.dists_p_inv),
    are_related_dists_a_inv = collect(df_synth.are_related_dists_a_inv)
)

model = fit(GeneralizedLinearModel, fx, df_collect, Bernoulli(), LogitLink())

data_orig = Tables.columntable(df_collect)
data = NamedTuple{keys(data_orig)}(map(copy, values(data_orig)))

compiled = compile_formula(model, data)
de = derivativeevaluator(:ad, compiled, data, vars)
J = Matrix{Float64}(undef, length(compiled), length(vars))

derivative_modelrow!(J, de, 1)
b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000
println("  Result: $(minimum(b).allocs) allocs, $(minimum(b).memory) bytes")

# Test: Manually reconstruct the computation
println("\n=== Test 3: Reconstruct synthetic data generation ===")
using Random
using Distributions

Random.seed!(08540)
n = 1000

# Generate base distances
dists_p = rand(LogNormal(1.3, 0.7), n)
are_related_dists_a = rand(LogNormal(1.8, 1.2), n)

# Compute inverses
dists_p_inv = 1 ./ dists_p
are_related_dists_a_inv = 1 ./ are_related_dists_a

# Handle Inf
finite_dists_p = filter(isfinite, dists_p_inv)
finite_are_related = filter(isfinite, are_related_dists_a_inv)

if !isempty(finite_dists_p)
    replace!(dists_p_inv, Inf => maximum(finite_dists_p))
end
if !isempty(finite_are_related)
    replace!(are_related_dists_a_inv, Inf => maximum(finite_are_related))
end

df_reconstructed = DataFrame(
    response = rand(Bernoulli(0.35), n),
    dists_p_inv = dists_p_inv,
    are_related_dists_a_inv = are_related_dists_a_inv
)

println("Reconstructed data statistics:")
println("  dists_p_inv min: ", minimum(df_reconstructed.dists_p_inv))
println("  dists_p_inv max: ", maximum(df_reconstructed.dists_p_inv))

model = fit(GeneralizedLinearModel, fx, df_reconstructed, Bernoulli(), LogitLink())

data_orig = Tables.columntable(df_reconstructed)
data = NamedTuple{keys(data_orig)}(map(copy, values(data_orig)))

compiled = compile_formula(model, data)
de = derivativeevaluator(:ad, compiled, data, vars)
J = Matrix{Float64}(undef, length(compiled), length(vars))

derivative_modelrow!(J, de, 1)
b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000
println("  Result: $(minimum(b).allocs) allocs, $(minimum(b).memory) bytes")

# Test: Skip the replace! step
println("\n=== Test 4: Without replace!() step ===")
Random.seed!(08540)

dists_p = rand(LogNormal(1.3, 0.7), n)
are_related_dists_a = rand(LogNormal(1.8, 1.2), n)

dists_p_inv_no_replace = 1 ./ dists_p
are_related_dists_a_inv_no_replace = 1 ./ are_related_dists_a

# Check for Inf
println("  Any Inf in dists_p_inv: ", any(isinf, dists_p_inv_no_replace))
println("  Any Inf in are_related_dists_a_inv: ", any(isinf, are_related_dists_a_inv_no_replace))

if !any(isinf, dists_p_inv_no_replace) && !any(isinf, are_related_dists_a_inv_no_replace)
    df_no_replace = DataFrame(
        response = rand(Bernoulli(0.35), n),
        dists_p_inv = dists_p_inv_no_replace,
        are_related_dists_a_inv = are_related_dists_a_inv_no_replace
    )

    model = fit(GeneralizedLinearModel, fx, df_no_replace, Bernoulli(), LogitLink())

    data_orig = Tables.columntable(df_no_replace)
    data = NamedTuple{keys(data_orig)}(map(copy, values(data_orig)))

    compiled = compile_formula(model, data)
    de = derivativeevaluator(:ad, compiled, data, vars)
    J = Matrix{Float64}(undef, length(compiled), length(vars))

    derivative_modelrow!(J, de, 1)
    b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000
    println("  Result: $(minimum(b).allocs) allocs, $(minimum(b).memory) bytes")
else
    println("  Skipped (contains Inf)")
end
