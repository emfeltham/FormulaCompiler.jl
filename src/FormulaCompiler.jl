module FormulaCompiler

################################ Dependencies ################################

# used only for testing
using Random
using Test

# true deps
using Dates: now
using Statistics
using StatsModels, GLM, CategoricalArrays, Tables, DataFrames, Random

import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel

using StandardizedPredictors: ZScoredTerm

using LinearAlgebra: dot, I
using ForwardDiff

using Base.Iterators: product # -> compute_kronecker_pattern

# useful for booleans in formulas
not(x::Bool) = !x
# N.B., this is dangerous -- does not clearly fail when x outside [0,1]
not(x::T) where {T<:Real} = one(x) - x
export not

################################# Evaluation #################################

# Include files in dependency order
include("fixed_helpers.jl") # No dependencies
include("evaluators.jl")
include("compile_term.jl")

# Main compilation interface
include("CompiledFormula.jl") # Clean execution plan system
export compile_formula

################################# Core system #################################

include("apply_function.jl")
include("get_data_value_specialized.jl")

include("step1_specialized_core.jl")
include("step2_categorical_support.jl")

include("step3_functions.jl")

# step 4

# include("phase1_interaction_positions.jl")
# include("phase2_interaction_allocator.jl")
# include("phase3_interaction_execution.jl")
# include("phase4_interaction_analysis.jl")
# include("phase5_7_integration_validation_testin.jl")
include("step4_interactions.jl")

export test_new_interaction_system

include("modelrow.jl")
export ModelRowEvaluator, modelrow!, modelrow

################################## Overrides ##################################

# include("override_1.jl")
export OverrideVector, create_categorical_override
export DataScenario, create_scenario, create_override_data, create_override_vector
# include("override_2.jl")
# include("override_4.jl")

############################## Derivative system ##############################

# include("derivative_step1_foundation.jl")
# export compile_derivative_formula
# include("derivative_step2_functions.jl")

export modelrow!, modelrow

end # end module
