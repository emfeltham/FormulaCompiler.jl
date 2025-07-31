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
    # include("test_step1_specialized_core.jl")
    # include("test_step2_categorical_support.jl")
    # include("test_step3_function_support.jl")
    # include("test_step4_interactions.jl")

    # include("test_position_mapping.jl")
    # include("step4_run_profiling.jl")
    # include("test_modelrow.jl")
    
    # Models
    include("test_models.jl")

    # Derivatives
    include("test_derivative_phase1.jl")
    include("test_derivative_phase2.jl")
    
    # Integration and compatibility tests
    # include("test_integration.jl")
    # include("test_mixed_models.jl")
    
    # Performance and regression tests
    # include("test_performance.jl") these may not be right, and are non-essential
    # include("test_regression.jl")
end
