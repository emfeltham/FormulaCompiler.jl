# test_cases.jl

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

basic = [
    (@formula(response ~ 1), "Baseline (no interactions)"),
    (@formula(response ~ x), "Baseline (no interactions)")
];

categoricals = [
    (@formula(response ~ group3), "Single categorical"),
    (@formula(response ~ group3 + group4), "Two categoricals"),
    (@formula(response ~ group3 + group4 + binary), "Three categoricals"),
    (@formula(response ~ x + group3), "Mixed continuous + categorical"),
    (@formula(response ~ x + y + group3 + group4), "Multiple mixed"),
];

functions = [
    (@formula(response ~ log(z)), "Function 1"),
    (@formula(response ~ z > 0), "Comparison 1"),
    (@formula(response ~ x^2), "Function 2"),
    (@formula(response ~ log(abs(z))), "2-Nested Function 1"),
    (@formula(response ~ abs(z)^2), "2-Nested Function 2"),
    (@formula(response ~ abs(log(abs(z)))), "3-Nested Function"),
    (@formula(response ~ log(abs(z))^2), "3-Nested Function 2"),
    (@formula(response ~ (z + y)^2), "2-Function"),
    (@formula(response ~ abs(z + x)), "2-Function 1"),
    (@formula(response ~ abs(z + y + x)), "3-Function 1"),
    (@formula(response ~ (z + y + x)^2), "3-Function 2"),
    (@formula(response ~ abs(z + y + x + w)), "3-Function 1")
];

interactions = [
    (@formula(response ~ x * y), "Simple 2-way interaction"),
    (@formula(response ~ x * group3), "Continuous × Categorical"),
    (@formula(response ~ group3 * binary), "Categorical × Categorical"),
    (@formula(response ~ log(z) * group4), "Function × Categorical"),
    (@formula(response ~ x * log(z)), "Continuous × Function"),

    (@formula(response ~ x * y * z), "3-way continuous"),
    (@formula(response ~ x * y * group3), "3-way interaction"),
    (@formula(response ~ group3 * group4 * binary), "3-way categorical"),
    
    (@formula(response ~ x * y * z * w), "4-way interaction"),
    (@formula(response ~ log(z) * exp(w) * group3), "Multiple functions × categorical"),
    (@formula(response ~ log(abs(x) + 2) * (y - 3.5)), "Complex Function 1"),
    (@formula(response ~ x * y * group3 + log(z) * group4), "Your original formula!"),
];

function test_cases(cases, df, data)
    if typeof(cases) <: Tuple
        cases = [cases]
    end
    println("Model scenario testing")
    for (f, nm) in cases
        # fit
        model = fit(LinearModel, f, df);
        # allocate output
        output_after = Vector{Float64}(undef, size(modelmatrix(model), 2));
        # compile
        compiled_after = compile_formula_specialized(model, data);
        compiled_after(output_after, data, 1);

        println("Case: " * nm)
        @btime $compiled_after($output_after, $data, $1);
    end
end

function test_correctness(cases, df, data)
    # normalize to a Vector
    cases = isa(cases, Tuple) ? [cases] : collect(cases)

    @testset "Model-scenario testing" begin
        for (f, nm) in cases
            @testset "$nm" begin
                # fit the model
                model = fit(LinearModel, f, df)
                mm    = modelmatrix(model)
                # prepare your “after” vector
                output_after = Vector{Float64}(undef, size(mm, 2))
                # compile
                compiled_after = compile_formula_specialized(model, data)
                # run it
                compiled_after(output_after, data, 1)
                # now the actual test
                @test isapprox(mm[1, :], output_after; atol = 1e-5)
            end
        end
    end
end

df, data = test_data(; n = 200);
x = functions
# x = interactions;
test_cases(x, df, data);

test_correctness(x, df, data)
