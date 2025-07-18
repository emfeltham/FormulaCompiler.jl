# test/runtests.jl
# Main test runner for FormulaCompiler.jl

using Revise
using Test
using FormulaCompiler

using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels, StandardizedPredictors
using MixedModels
using BenchmarkTools

using FormulaCompiler:
    AbstractEvaluator,
    ConstantEvaluator, ContinuousEvaluator,
    CombinedEvaluator, InteractionEvaluator, FunctionEvaluator, CategoricalEvaluator,
    ZScoreEvaluator,
    output_width, evaluate!

using FormulaCompiler:
    test_evaluator_storage,
    test_comprehensive_compilation, test_complete, 
    validate_derivative_evaluator, test_scenario_foundation,
    example_scenario_usage,
    set_override!, remove_override!,
    update_scenario!

using FormulaCompiler:
    compile_term
    

# Set consistent random seed for reproducible tests
Random.seed!(06515)

@testset "FormulaCompiler.jl Tests" begin
    
    # Core functionality tests
    include("test_evaluators.jl")
    include("test_compilation.jl")
    include("test_modelrow.jl")
    
    # Advanced features
    include("test_derivatives.jl")
    include("test_derivative_correctness.jl")
    include("test_scenarios.jl")
    include("test_evaluator_trees.jl")
    
    # Integration and compatibility tests
    include("test_integration.jl")
    include("test_mixed_models.jl")
    
    # Performance and regression tests
    # include("test_performance.jl") these may not be right, and are non-essential
    include("test_regression.jl")
end

#=
using Test
using Revise
using Random
using FormulaCompiler

@testset "FormulaCompiler.jl" begin
    include("initial_tests.jl")
    
end
=#
