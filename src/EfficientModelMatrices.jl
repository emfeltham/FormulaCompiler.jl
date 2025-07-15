module EfficientModelMatrices

# ============================================================================
# Deps.
# ============================================================================

using Random # testing?

using StatsModels, GLM, CategoricalArrays, Tables, DataFrames, Random

import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel

# Support for StandardizedPredictors.jl
using StandardizedPredictors: ZScoredTerm

# Include files in dependency order
include("fixed_helpers.jl")     # No dependencies
include("evaluators.jl")        # Uses fixed_helpers
include("generators.jl")        # Uses evaluators + fixed_helpers
include("testing.jl")

# using LinearAlgebra

# useful for booleans in formulas
not(x::Bool) = !x
not(x::T) where {T<:Real} = one(x) - x
export not

export fixed_effects_form
export compile_formula, CompiledFormula, test_complete

end # module
