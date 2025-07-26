module FormulaCompiler

# ============================================================================
# Deps.
# ============================================================================

using Random # testing
using Test
using Statistics
using ForwardDiff
using StatsModels, GLM, CategoricalArrays, Tables, DataFrames, Random

import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel

# Support for StandardizedPredictors.jl
using StandardizedPredictors: ZScoredTerm

# useful for booleans in formulas
not(x::Bool) = !x
# N.B., this is dangerous -- does not clearly fail when x outside [0,1]
not(x::T) where {T<:Real} = one(x) - x
export not

# Include files in dependency order
include("fixed_helpers.jl")     # No dependencies
export fixed_effects_form
include("evaluators.jl")        # Uses fixed_helpers


"""
    ExecutionBlock

Abstract base type for different kinds of execution blocks.
Each block represents a group of operations that can be executed together.
"""
abstract type ExecutionBlock end

"""
    ValidatedExecutionPlan

The only execution plan type. Construction validates everything once,
execution is guaranteed zero-allocation.
"""
struct ValidatedExecutionPlan
    scratch_size::Int
    blocks::Vector{ExecutionBlock}
    total_output_width::Int
    data_length::Int                    # Cached for bounds checking
    validated_columns::Set{Symbol}     # Columns guaranteed to exist
    
    function ValidatedExecutionPlan(evaluator::AbstractEvaluator, data::NamedTuple)
        # Generate the execution plan structure
        basic_plan = generate_execution_plan_structure(evaluator)
        
        # Comprehensive validation with helpful error messages
        validate_plan_against_data!(basic_plan, data)
        
        # Cache information for zero-allocation execution
        data_length = length(first(data))
        validated_columns = Set(keys(data))
        
        new(basic_plan.scratch_size, basic_plan.blocks, basic_plan.total_output_width, 
            data_length, validated_columns)
    end
end

include("CompiledFormula.jl")   # Defines key structs and methods - NOW INCLUDES DERIVATIVES
export compile_formula, CompiledFormula, test_complete
# Derivatives
export compile_derivative_formula, CompiledDerivativeFormula
export clear_derivative_cache!, list_compiled_derivatives

# include("evaluator_trees.jl")
export extract_root_evaluator, get_evaluator_tree, has_evaluator_access
export count_evaluator_nodes, get_variable_dependencies, get_evaluator_summary
export print_evaluator_tree, test_evaluator_storage
# include("generators.jl")        # Uses evaluators + fixed_helpers
export generate_code_from_evaluator, generate_evaluator_code!
export test_phase2a_complete, test_phase2a_architecture
export generate_expression_recursive, generate_statements_recursive


include("modelrow!.jl")
include("modelrow.jl")
# export modelrow!
export modelrow
export clear_model_cache!, test_modelrow_interface
export ModelRowEvaluator

include("override.jl")
export OverrideVector, create_categorical_override
export DataScenario, create_scenario, create_override_data, create_override_vector
export ScenarioCollection, create_scenario_grid, create_scenario_combinations
export get_scenario_by_name, list_scenarios
export modelrow!, modelrow_scenarios!
export test_scenario_foundation, example_scenario_usage

include("derivative_evaluators.jl")
include("CompiledDerivativeFormula.jl")   # Defines key structs and methods - NOW INCLUDES DERIVATIVES
include("derivative_generators.jl")
include("derivative_modelrow.jl")
export compute_derivative_evaluator, compute_interaction_derivative_recursive
export compute_nary_product_derivative, compute_division_derivative, compute_power_derivative
export marginal_effects!

export ScaledEvaluator, ProductEvaluator
export ChainRuleEvaluator, ProductRuleEvaluator, ForwardDiffEvaluator
export get_standard_derivative_function, is_zero_derivative, validate_derivative_evaluator

include("testing.jl")
include("testing_derivatives.jl")

include("phase_tests.jl")

## alt approach

include("ast_decomposition.jl")
include("execution_plans.jl")
include("test_updated_execution_plans.jl")

end # module
