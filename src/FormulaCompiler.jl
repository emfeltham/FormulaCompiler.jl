module FormulaCompiler

# ============================================================================
# Deps.
# ============================================================================

 # testing
using Random
using Test

# true deps
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
include("fixed_helpers.jl") # No dependencies
export fixed_effects_form


using Base.Iterators: product # -> compute_kronecker_pattern

include("evaluators.jl") # Uses fixed_helpers
# Core evaluator system exports
export AbstractEvaluator, ConstantEvaluator, ContinuousEvaluator, CategoricalEvaluator
export FunctionEvaluator, InteractionEvaluator, ZScoreEvaluator, CombinedEvaluator
export ScaledEvaluator, ProductEvaluator
export output_width, evaluate!, compile_term, extract_all_columns

# include("assign_names.jl")

# include("execution.jl") # OLD
include("execute_self_contained.jl")
include("execute_to_scratch.jl")
include("compile_term.jl")
export test_self_contained_evaluators, compile_term, execute_self_contained!
export create_execution_plan, generate_blocks!


# Main compilation interface
include("CompiledFormula.jl") # Clean execution plan system
export CompiledFormula, compile_formula
export get_scratch_size, get_column_names, get_evaluator_tree
export show_execution_plan, benchmark_execution, is_zero_allocation

# Row evaluation interfaces
# include("matrix_writer.jl")
include("modelrow!.jl")
export modelrow!, test_updated_modelrow_system
include("modelrow.jl")
export modelrow
export clear_model_cache!
export ModelRowEvaluator

# Override and scenario system
# include("override.jl")
# export OverrideVector, create_categorical_override
# export DataScenario, create_scenario, create_override_data, create_override_vector
# export ScenarioCollection, create_scenario_grid, create_scenario_combinations
# export get_scenario_by_name, list_scenarios
# export modelrow!, modelrow_scenarios!

# Derivative system
# include("derivative_evaluators.jl")
# include("CompiledDerivativeFormula.jl")   
# include("derivative_generators.jl")
# include("derivative_modelrow.jl")
# export compile_derivative_formula, CompiledDerivativeFormula
# export clear_derivative_cache!, list_compiled_derivatives
# export compute_derivative_evaluator, compute_interaction_derivative_recursive
# export compute_nary_product_derivative, compute_division_derivative, compute_power_derivative
# export marginal_effects!
# export ChainRuleEvaluator, ProductRuleEvaluator, ForwardDiffEvaluator
# export get_standard_derivative_function, is_zero_derivative, validate_derivative_evaluator

include("step1_specialized_core.jl")
include("step2_categorical_support.jl")
include("step3_function_support.jl")
include("step3_polish_linear.jl")
include("step4_interactions.jl")

end # end module
