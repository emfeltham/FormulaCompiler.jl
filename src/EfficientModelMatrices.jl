module EfficientModelMatrices

using Tables
using CategoricalArrays

import StatsModels
import StatsModels: 
    width, FormulaTerm, FunctionTerm, InteractionTerm,
    MatrixTerm, ConstantTerm, ContinuousTerm,
    CategoricalTerm, InterceptTerm

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


end # module