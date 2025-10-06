"""
Test to understand what causes allocations with synthetic data.
The hypothesis that NamedTuple size causes allocations doesn't make sense for per-row operations.
Let's dig deeper into the actual data structure.
"""

using BenchmarkTools
using GLM
using DataFrames
using Tables
using FormulaCompiler
using FormulaCompiler: derivativeevaluator, derivative_modelrow!

include("../test/support/generate_large_synthetic_data.jl")

println("Investigating data structure allocations...\n")

# Generate synthetic data
df_synth = generate_synthetic_dataset(1000; seed=08540)

# Simple test formula
fx = @formula(response ~ dists_p_inv + are_related_dists_a_inv)

# Fit model with synthetic data
model_synth = fit(GeneralizedLinearModel, fx, df_synth, Bernoulli(), LogitLink())

# Test 1: Original synthetic data converted to columntable
println("="^80)
println("Test 1: Synthetic data -> Tables.columntable -> copy")
println("="^80)

data_synth_orig = Tables.columntable(df_synth)
data_synth = NamedTuple{keys(data_synth_orig)}(map(copy, values(data_synth_orig)))

println("Type of data_synth: ", typeof(data_synth))
println("Type of response: ", typeof(data_synth.response))
println("Type of dists_p_inv: ", typeof(data_synth.dists_p_inv))

compiled_synth = compile_formula(model_synth, data_synth)
de_synth = derivativeevaluator(:ad, compiled_synth, data_synth, [:dists_p_inv, :are_related_dists_a_inv])
J_synth = Matrix{Float64}(undef, length(compiled_synth), 2)

derivative_modelrow!(J_synth, de_synth, 1)
b_synth = @benchmark derivative_modelrow!($J_synth, $de_synth, 1) samples=1000

println("Allocations: $(minimum(b_synth).allocs)")
println("Memory: $(minimum(b_synth).memory) bytes\n")

# Test 2: Create completely fresh DataFrame with same values
println("="^80)
println("Test 2: Fresh DataFrame with SAME values")
println("="^80)

df_fresh = DataFrame(
    response = Vector{Bool}(df_synth.response),
    dists_p_inv = Vector{Float64}(df_synth.dists_p_inv),
    are_related_dists_a_inv = Vector{Float64}(df_synth.are_related_dists_a_inv)
)

model_fresh = fit(GeneralizedLinearModel, fx, df_fresh, Bernoulli(), LogitLink())

data_fresh_orig = Tables.columntable(df_fresh)
data_fresh = NamedTuple{keys(data_fresh_orig)}(map(copy, values(data_fresh_orig)))

println("Type of data_fresh: ", typeof(data_fresh))
println("Type of response: ", typeof(data_fresh.response))
println("Type of dists_p_inv: ", typeof(data_fresh.dists_p_inv))

compiled_fresh = compile_formula(model_fresh, data_fresh)
de_fresh = derivativeevaluator(:ad, compiled_fresh, data_fresh, [:dists_p_inv, :are_related_dists_a_inv])
J_fresh = Matrix{Float64}(undef, length(compiled_fresh), 2)

derivative_modelrow!(J_fresh, de_fresh, 1)
b_fresh = @benchmark derivative_modelrow!($J_fresh, $de_fresh, 1) samples=1000

println("Allocations: $(minimum(b_fresh).allocs)")
println("Memory: $(minimum(b_fresh).memory) bytes\n")

# Test 3: Check what Tables.columntable returns
println("="^80)
println("Test 3: Inspect Tables.columntable behavior")
println("="^80)

ct_synth = Tables.columntable(df_synth)
ct_fresh = Tables.columntable(df_fresh)

println("Synthetic columntable type: ", typeof(ct_synth))
println("Fresh columntable type: ", typeof(ct_fresh))

println("\nSynthetic response type: ", typeof(ct_synth.response))
println("Fresh response type: ", typeof(ct_fresh.response))

println("\nAre they the same? ", typeof(ct_synth.response) == typeof(ct_fresh.response))

# Test 4: What if we don't copy at all?
println("\n" * "="^80)
println("Test 4: Use Tables.columntable WITHOUT copying")
println("="^80)

data_synth_nocopy = Tables.columntable(df_synth)
data_fresh_nocopy = Tables.columntable(df_fresh)

println("Testing synthetic (no copy)...")
compiled_synth_nc = compile_formula(model_synth, data_synth_nocopy)
de_synth_nc = derivativeevaluator(:ad, compiled_synth_nc, data_synth_nocopy, [:dists_p_inv, :are_related_dists_a_inv])
J_synth_nc = Matrix{Float64}(undef, length(compiled_synth_nc), 2)

derivative_modelrow!(J_synth_nc, de_synth_nc, 1)
b_synth_nc = @benchmark derivative_modelrow!($J_synth_nc, $de_synth_nc, 1) samples=1000
println("  Allocations: $(minimum(b_synth_nc).allocs)")
println("  Memory: $(minimum(b_synth_nc).memory) bytes")

println("\nTesting fresh (no copy)...")
compiled_fresh_nc = compile_formula(model_fresh, data_fresh_nocopy)
de_fresh_nc = derivativeevaluator(:ad, compiled_fresh_nc, data_fresh_nocopy, [:dists_p_inv, :are_related_dists_a_inv])
J_fresh_nc = Matrix{Float64}(undef, length(compiled_fresh_nc), 2)

derivative_modelrow!(J_fresh_nc, de_fresh_nc, 1)
b_fresh_nc = @benchmark derivative_modelrow!($J_fresh_nc, $de_fresh_nc, 1) samples=1000
println("  Allocations: $(minimum(b_fresh_nc).allocs)")
println("  Memory: $(minimum(b_fresh_nc).memory) bytes")

# Test 5: Check if the issue is in the derivativeevaluator itself
println("\n" * "="^80)
println("Test 5: Compare derivativeevaluator types")
println("="^80)

println("Synthetic DE type: ", typeof(de_synth_nc))
println("\nFresh DE type: ", typeof(de_fresh_nc))
println("\nSame type? ", typeof(de_synth_nc) == typeof(de_fresh_nc))
