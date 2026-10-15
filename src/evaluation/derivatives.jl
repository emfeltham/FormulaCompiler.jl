# derivatives.jl - Main interface for derivative computations
#
# High-level public API and exports for the derivative system.
# The implementation is organized into focused modules in the derivatives/ subdirectory.

# Load all derivative system modules
include("derivatives/types.jl")
include("derivatives/evaluator.jl")
include("derivatives/automatic_diff.jl")
include("derivatives/finite_diff.jl")
# REMOVED (2025-10-07): Marginal effects functions migrated to Margins.jl v2.0
# - marginal_effects.jl (wrapper functions)
# - marginal_effects_automatic_diff.jl (AD backend implementations)
# - marginal_effects_finite_diff.jl (FD backend implementations)
# - gradients.jl (delta_method_se)
include("derivatives/contrasts.jl")
include("derivatives/link_functions.jl")

# Export public API (the marginal-effects, gradient and delta-method functions formerly
# listed here moved to Margins.jl in v2.0; see src/FormulaCompiler.jl)
export
       derivativeevaluator,
       derivative_modelrow!, derivative_modelrow,
       contrast_modelrow!, contrast_modelrow,
       continuous_variables,
       fd_jacobian_column!
