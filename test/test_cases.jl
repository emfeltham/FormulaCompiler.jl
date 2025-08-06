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

###############################################################################
# RUN TESTS
###############################################################################

df, data = test_data(; n = 200);

cases = [test_scenarios.basic..., test_scenarios.categoricals..., test_scenarios.functions..., test_scenarios.interactions...];

# cases = functions
test_cases(cases, df, data);

println("___________________________________________________________")

test_correctness(cases, df, data);

# manually check a particular ouput
# cases = interactions;

# let j = 13
#     i = 1
    
#     f, nm = cases[j]
#     f, nm = (@formula(response ~ x & group3 + x), "Continuous × Categorical 2")
#     model = fit(LinearModel, f, df)
#     mm = modelmatrix(model);
#     mr = mm[i, :]
#     # prepare your “after” vector
#     output_after = Vector{Float64}(undef, size(mm, 2))
#     # compile
#     compiled_after = compile_formula(model, data);
#     # run it
#     compiled_after(output_after, data, i)
#     # now the actual test
#     @show hcat(coefnames(model), mr, output_after)
#     @test isapprox(mr, output_after; atol = 1e-5)
# end

# ix = findall(.!(mm[1, :] .== output_after))
# coefnames(model)[ix]

# hcat(mm[1, :], output_after)
