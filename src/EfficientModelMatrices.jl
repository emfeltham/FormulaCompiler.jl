module EfficientModelMatrices

# ============================================================================
# Deps.
# ============================================================================

using Random # testing
using Test

using StatsModels, GLM, CategoricalArrays, Tables, DataFrames, Random

import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel

# Support for StandardizedPredictors.jl
using StandardizedPredictors: ZScoredTerm

# Include files in dependency order
include("fixed_helpers.jl")     # No dependencies
include("evaluators.jl")        # Uses fixed_helpers
include("generators.jl")        # Uses evaluators + fixed_helpers

# useful for booleans in formulas
not(x::Bool) = !x
# N.B., this is dangerous -- does not clearly fail when x outside [0,1]
not(x::T) where {T<:Real} = one(x) - x
export not

export fixed_effects_form
export compile_formula, CompiledFormula, test_complete

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

end # module
