module EfficientModelMatrices

using LinearAlgebra

using Tables
using CategoricalArrays

import StatsModels
import StatsModels:
    formula, @formula,
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

# misc background methods
include("data_validation.jl")
include("termmapping.jl")
include("termmapping_add.jl")

# ============================================================================
# Core API: Basic efficient model matrix construction
# ============================================================================


# main workflow
include("structure_analysis.jl")

export analyze_formula_structure, FormulaAnalysis, TermAnalysis
export validate_analysis, print_analysis_summary, test_structure_analysis

include("instruction_generation.jl")
export generate_instructions, generate_term_instructions, test_instruction_generation
export test_interaction_fix

include("generated_integration.jl")
export compile_formula_complete, modelrow!, get_compiled_function
export test_compilation_performance, test_complete_pipeline

include("testing_suite.jl")

end # module
