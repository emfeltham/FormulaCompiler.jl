# test_step4_interactions.jl

using Revise
using Test
using BenchmarkTools, Profile
using FormulaCompiler

using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels, StandardizedPredictors
using MixedModels
using BenchmarkTools
    
# Set consistent random seed for reproducible tests
Random.seed!(06515)

using FormulaCompiler:
    compile_formula_specialized

###############################################################################
# TESTING FUNCTIONS
###############################################################################

function test_data(; n = 200)
    # Create test data with all variable types
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,  # Positive for log
        w = randn(n),
        t = randn(n),
        group3 = categorical(rand(["A", "B", "C"], n)),           
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      
        binary = categorical(rand(["Yes", "No"], n)),             
        group5 = categorical(rand(["P", "Q", "R", "S", "T"], n)), 
        response = randn(n)
    )
    data = Tables.columntable(df)
    return df, data
end

test_cases = [
    (@formula(response ~ 1), "Baseline (no interactions)"),
    (@formula(response ~ x), "Baseline (no interactions)"),
    (@formula(response ~ log(z)), "Function"),
    (@formula(response ~ group3), "Single categorical"),
    (@formula(response ~ group3 + group4), "Two categoricals"),
    (@formula(response ~ group3 + group4 + binary), "Three categoricals"),
    (@formula(response ~ x + group3), "Mixed continuous + categorical"),
    (@formula(response ~ x + y + group3 + group4), "Multiple mixed"),
    (@formula(response ~ x * y), "Simple 2-way interaction"),
    (@formula(response ~ x * group3), "Continuous × Categorical"),
    (@formula(response ~ group3 * binary), "Categorical × Categorical"),
    (@formula(response ~ log(z) * group4), "Function × Categorical"),
    (@formula(response ~ x * y * group3), "3-way interaction"),
    (@formula(response ~ x * log(z)), "Continuous × Function"),
    (@formula(response ~ x * y * group3 + log(z) * group4), "Your original formula!"),
    (@formula(response ~ x * y * z), "3-way continuous"),
    (@formula(response ~ group3 * group4 * binary), "3-way categorical"),
    (@formula(response ~ x * y * z * w), "4-way interaction"),
    (@formula(response ~ log(z) * exp(w) * group3), "Multiple functions × categorical"),
];

function test_step4_cases(test_cases, df, data)
    println("STEP 4 TESTING: COMPLETE SPECIALIZATION WITH INTERACTIONS")
    for (f, nm) in test_cases
        # fit
        model = fit(LinearModel, f, df);
        # allocate output
        output_after = Vector{Float64}(undef, size(modelmatrix(model), 2));
        # compile
        compiled_after = compile_formula_specialized(model, data);
        println("Case: " * nm)
        @btime $compiled_after($output_after, $data, $1);
    end
end

df, data = test_data(; n = 200);
test_step4_cases(test_cases, df, data);

