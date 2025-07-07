module EfficientModelMatrices

using Tables
using CategoricalArrays

import StatsModels
import StatsModels:
    formula,
    width,
    Term, FormulaTerm, FunctionTerm, InteractionTerm,
    MatrixTerm, ConstantTerm, ContinuousTerm,
    CategoricalTerm, InterceptTerm,
    StatisticalModel

using GLM: LinearModel, GeneralizedLinearModel
import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel

# ============================================================================
# Helpers
# ============================================================================

include("fixed_helpers.jl")
export fixed_effects_form

# ============================================================================
# Core Mechanics: Replace `modelcols`
# ============================================================================

include("InplaceModeler.jl")
include("_cols!.jl")
export InplaceModeler

# ============================================================================
# Core API: Basic efficient model matrix construction
# ============================================================================

include("modelmatrix!.jl")
export modelmatrix!, extract_model_matrix

include("data_validation.jl")

end # module