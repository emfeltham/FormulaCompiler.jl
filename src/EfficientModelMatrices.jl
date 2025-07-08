module EfficientModelMatrices

using Tables
using CategoricalArrays

import StatsModels
import StatsModels:
    formula,
    width,
    Term, AbstractTerm, FormulaTerm, FunctionTerm, InteractionTerm,
    MatrixTerm, ConstantTerm, ContinuousTerm,
    CategoricalTerm, InterceptTerm,
    width, termvars,
    StatisticalModel
using GLM: LinearModel, GeneralizedLinearModel
import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel

# Support for StandardizedPredictors.jl
using StandardizedPredictors: ZScoredTerm

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
include("standardized.jl")
export InplaceModeler

# ============================================================================
# Core API: Basic efficient model matrix construction
# ============================================================================

include("modelmatrix!.jl")
export modelmatrix!, extract_model_matrix

include("data_validation.jl")

include("termmapping.jl")
export ColumnMapping, build_column_mapping, build_enhanced_mapping, enhanced_column_mapping
export get_variable_ranges, get_all_variable_columns, get_term_for_column, get_terms_for_columns
export collect_termvars_recursive, evaluate_single_column!


end # module
