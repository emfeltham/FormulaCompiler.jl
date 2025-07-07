# runtests.jl

using Test
using Revise
using EfficientModelMatrices
using StatsModels, DataFrames, CategoricalArrays, Tables
using GLM, MixedModels, Random
using StandardizedPredictors
using Statistics

@testset "EfficientModelMatrices.jl" begin
    include("main_tests.jl")
    include("standardized_tests.jl")
end
