"""
Test if large NamedTuple passed to derivativeevaluator causes allocations.
"""

using BenchmarkTools
using GLM
using DataFrames
using Tables
using FormulaCompiler
using FormulaCompiler: derivativeevaluator, derivative_modelrow!

include("../test/support/generate_large_synthetic_data.jl")

println("Testing NamedTuple size impact on AD allocations...\n")

# Generate synthetic data (50+ columns)
df_full = generate_synthetic_dataset(1000; seed=08540)

# Extract only needed columns
df_minimal = DataFrame(
    response = df_full.response,
    dists_p_inv = df_full.dists_p_inv,
    are_related_dists_a_inv = df_full.are_related_dists_a_inv
)

println("Full dataframe: $(ncol(df_full)) columns")
println("Minimal dataframe: $(ncol(df_minimal)) columns\n")

# Test formula
fx = @formula(response ~ dists_p_inv + are_related_dists_a_inv)

for (name, df) in [("Full (50+ cols)", df_full), ("Minimal (3 cols)", df_minimal)]
    println("="^80)
    println("Testing: $name")
    println("="^80)

    # Fit model
    model = fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())

    # Convert to columntable and copy
    data_orig = Tables.columntable(df)
    println("  NamedTuple size: $(length(keys(data_orig))) fields")

    data = NamedTuple{keys(data_orig)}(map(copy, values(data_orig)))

    # Compile
    compiled = compile_formula(model, data)

    # Build derivative evaluator - THIS is where we pass the large NamedTuple
    vars = [:dists_p_inv, :are_related_dists_a_inv]
    de = derivativeevaluator(:ad, compiled, data, vars)
    J = Matrix{Float64}(undef, length(compiled), length(vars))

    # Warmup
    derivative_modelrow!(J, de, 1)

    # Benchmark
    b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000

    println("  Allocations: $(minimum(b).allocs)")
    println("  Memory: $(minimum(b).memory) bytes")
    println("  Time: $(minimum(b).time) ns")
    println()
end

# Test: Create minimal data_cols for derivativeevaluator but full for compile_formula
println("="^80)
println("Test: Compile with full, but pass minimal to derivativeevaluator")
println("="^80)

model = fit(GeneralizedLinearModel, fx, df_full, Bernoulli(), LogitLink())

# Full data for compilation
data_full_orig = Tables.columntable(df_full)
data_full = NamedTuple{keys(data_full_orig)}(map(copy, values(data_full_orig)))

# Minimal data for derivatives (only needed columns)
data_minimal_orig = Tables.columntable(df_minimal)
data_minimal = NamedTuple{keys(data_minimal_orig)}(map(copy, values(data_minimal_orig)))

compiled = compile_formula(model, data_full)
println("  Compiled with: $(length(keys(data_full))) fields")

# Pass MINIMAL data to derivativeevaluator
vars = [:dists_p_inv, :are_related_dists_a_inv]
de = derivativeevaluator(:ad, compiled, data_minimal, vars)
println("  Evaluator with: $(length(keys(data_minimal))) fields")

J = Matrix{Float64}(undef, length(compiled), length(vars))

derivative_modelrow!(J, de, 1)
b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000

println("  Allocations: $(minimum(b).allocs)")
println("  Memory: $(minimum(b).memory) bytes")
println("  Time: $(minimum(b).time) ns")
