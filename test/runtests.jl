# test/runtests.jl
# Main test runner for FormulaCompiler.jl

using Revise
using Test
using BenchmarkTools, Profile
using FormulaCompiler
using Random

using Statistics
using DataFrames, Tables, CategoricalArrays
using StatsModels, StandardizedPredictors
using MixedModels, GLM
    
# Set consistent random seed for reproducible tests
Random.seed!(06515)

@testset "FormulaCompiler.jl Tests" begin
    
    # Core functionality tests
    
    # these are dignostic rather than formal tests
    # include("test_position_mapping.jl")
    
    # Models
    include("test_allocations.jl") # performance
    include("test_models.jl") # correctness

end
