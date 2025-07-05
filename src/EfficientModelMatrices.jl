module EfficientModelMatrices

using StatsModels
using Tables
using DataFrames: nrow
using CategoricalArrays


# ============================================================================
# Core API: Basic efficient model matrix construction
# ============================================================================

include("modelmatrix!.jl")
export modelmatrix!

# ============================================================================
# Advanced API: Selective updates and dependency analysis
# ============================================================================

include("dependencies.jl")
include("selective_updates.jl")

export analyze_dependencies
export update_matrix_subset!
export estimate_term_width
export MatrixUpdatePlan

# ============================================================================
# Convenience API: High-level workflows
# ============================================================================

include("workflows.jl")

export incremental_update!
export create_update_plan

end # module