"""
Check what columns are actually needed by the compiled formula.
"""

using GLM
using DataFrames
using Tables
using FormulaCompiler
using BenchmarkTools

include("../test/support/generate_large_synthetic_data.jl")

# Generate synthetic data
df = generate_synthetic_dataset(1000; seed=08540)

# The complex formula
fx = @formula(response ~
    socio4 +
    (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
    !socio4 & dists_a_inv +
    # individual variables
    (schoolyears_p + wealth_d1_4_p + man_p + age_p + religion_c_p +
    same_building + population +
    hhi_religion + hhi_indigenous +
    coffee_cultivation + market +
    relation) & (1 + socio4 + are_related_dists_a_inv) +
    # tie variables
    (
        degree_a_mean + degree_h +
        age_a_mean + age_h * age_h_nb_1_socio +
        schoolyears_a_mean + schoolyears_h * schoolyears_h_nb_1_socio +
        man_x * man_x_mixed_nb_1 +
        wealth_d1_4_a_mean + wealth_d1_4_h * wealth_d1_4_h_nb_1_socio +
        isindigenous_x * isindigenous_homop_nb_1 +
        religion_c_x * religion_homop_nb_1
    ) & (1 + socio4 + are_related_dists_a_inv) +
    religion_c_x & hhi_religion +
    isindigenous_x & hhi_indigenous
)

model = fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())

# Get all column names referenced in the formula
println("Formula term names: ", coefnames(model))
println("\nTotal terms: ", length(coefnames(model)))

# Check if FormulaCompiler tracks which columns are actually used
data_cols = Tables.columntable(df)
compiled = compile_formula(model, data_cols)

println("\nChecking if we can determine required columns...")
println("Type of compiled: ", typeof(compiled))

# The formula references these columns
formula_vars = Set([
    :response, :socio4, :dists_p_inv, :are_related_dists_a_inv, :dists_a_inv,
    :schoolyears_p, :wealth_d1_4_p, :man_p, :age_p, :religion_c_p,
    :same_building, :population, :hhi_religion, :hhi_indigenous,
    :coffee_cultivation, :market, :relation,
    :degree_a_mean, :degree_h, :age_a_mean, :age_h, :age_h_nb_1_socio,
    :schoolyears_a_mean, :schoolyears_h, :schoolyears_h_nb_1_socio,
    :man_x, :man_x_mixed_nb_1, :wealth_d1_4_a_mean, :wealth_d1_4_h,
    :wealth_d1_4_h_nb_1_socio, :isindigenous_x, :isindigenous_homop_nb_1,
    :religion_c_x, :religion_homop_nb_1
])

println("\nColumns referenced in formula: ", length(formula_vars))
println("Total columns in df: ", ncol(df))

# Test if passing only formula columns works
data_cols_formula_only = NamedTuple{Tuple(sort(collect(formula_vars)))}(
    Tuple(data_cols[col] for col in sort(collect(formula_vars)))
)

println("\nTesting with formula-only columns ($(length(keys(data_cols_formula_only))) cols)...")
try
    compiled_minimal = compile_formula(model, data_cols_formula_only)
    println("✓ Successfully compiled with $(length(keys(data_cols_formula_only))) columns")

    # Test derivatives
    using FormulaCompiler: derivativeevaluator, derivative_modelrow!
    vars = [:age_h, :dists_p_inv, :are_related_dists_a_inv, :schoolyears_h]
    de = derivativeevaluator(:ad, compiled_minimal, data_cols_formula_only, vars)
    J = Matrix{Float64}(undef, length(compiled_minimal), length(vars))

    derivative_modelrow!(J, de, 1)

    b = @benchmark derivative_modelrow!($J, $de, 1) samples=1000
    println("  Allocations: $(minimum(b).allocs)")
    println("  Memory: $(minimum(b).memory) bytes")

catch e
    println("✗ Failed: ", e)
end
