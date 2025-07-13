module EfficientModelMatrices

using LinearAlgebra

using Tables
using CategoricalArrays

import StatsModels
import StatsModels:
    formula,
    width, hasintercept,
    Term, AbstractTerm, FormulaTerm, FunctionTerm, InteractionTerm,
    MatrixTerm, ConstantTerm, ContinuousTerm,
    CategoricalTerm, InterceptTerm,
    width, termvars,
    StatisticalModel,
    modelmatrix
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

include("ColumnMapping.jl")
include("InplaceModeler.jl")
export InplaceModeler

# include("_cols2.jl")
# include("standardized.jl")

# ============================================================================
# Core API: Basic efficient model matrix construction
# ============================================================================

# include("modelmatrix!.jl")
# export modelmatrix!, extract_model_matrix

include("data_validation.jl")

include("termmapping.jl")

export ColumnMapping, build_column_mapping, build_enhanced_mapping, enhanced_column_mapping
export get_variable_ranges, get_all_variable_columns, get_term_for_column, get_terms_for_columns
export collect_termvars_recursive, evaluate_single_column!

include("termmapping_add.jl")
export 
    get_variable_columns_flat, get_unchanged_columns,
    build_perturbation_plan,
    validate_column_mapping, get_variable_term_ranges,
    build_variable_term_map

include("compiled_formula.jl")
include("compiled_formula_generated.jl")

end # module
