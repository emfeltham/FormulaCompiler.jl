module FormulaCompiler

################################ Dependencies ################################

# Development dependencies (remove from production builds)
using Random, Test, BenchmarkTools

# Core dependencies
using Dates: now
using Statistics
using StatsModels, GLM, CategoricalArrays, Tables, DataFrames
using LinearAlgebra: dot, I
using ForwardDiff
using Base.Iterators: product # -> compute_kronecker_pattern

# External package integration
import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel
using StandardizedPredictors: ZScoredTerm

################################# Core System #################################

# Core utilities and types
include("core/utilities.jl")
export not, OverrideVector

################################# Compilation #################################

# External package integration
include("integration/mixed_models.jl")

# Evaluation system (needed by compilation)
include("evaluation/evaluators.jl")
include("evaluation/data_access.jl")
include("evaluation/function_ops.jl")

# Compilation system
include("compilation/term_compiler.jl")
include("compilation/legacy_compiled.jl")
export compile_formula_

# Specialized compilation pipeline
include("compilation/pipeline/step1_constants.jl")
include("compilation/pipeline/step2_categorical.jl")
include("compilation/pipeline/step3_functions.jl")
include("compilation/pipeline/step4_interactions.jl")
include("compilation/pipeline/step4_function_interactions.jl")

export test_new_interaction_system, compile_formula

################################# Evaluation #################################

# High-level evaluation interface
include("evaluation/modelrow.jl")
export ModelRowEvaluator, modelrow!, modelrow

################################## Scenarios ##################################

# Override and scenario system
include("scenarios/overrides.jl")
export create_categorical_override, create_scenario_grid
export DataScenario, create_scenario, create_override_data, create_override_vector

############################## Development Tools ##############################

# Development utilities (only include in dev builds)
include("dev/testing_utilities.jl")

############################## Future Features ##############################

# Derivative system (under development)
# include("derivatives/step1_foundation.jl")
# include("derivatives/step2_functions.jl")
# export compile_derivative_formula

end # end module
