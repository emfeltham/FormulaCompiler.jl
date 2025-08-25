# test/runtests.jl
# Main test runner for FormulaCompiler.jl
# julia --project="." test/runtests.jl > test/tests.txt 2>&1

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
    
    # Core functionality
    include("test_position_mapping.jl") # Position mapping system
    
    # Models
    include("test_allocations.jl") # performance
    include("test_models.jl") # correctness

end
