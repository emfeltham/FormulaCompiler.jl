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

# Export public API
export 
       derivativeevaluator,
       derivative_modelrow!, derivative_modelrow,
       derivative_modelrow_fd,
       # High-performance direct function calls
       marginal_effects_eta!, marginal_effects_eta!,
       marginal_effects_mu!, marginal_effects_mu!,
       marginal_effects_eta_grad!, marginal_effects_eta_grad,
       # Simple concrete type dispatch functions (Step 3.5.1)
       marginal_effects_eta!, marginal_effects_mu!,
       contrast_modelrow!, contrast_modelrow,
       continuous_variables,
       # Single-column FD and parameter gradient functions
       fd_jacobian_column!, fd_jacobian_column_pos!,
       me_eta_grad_beta!,
       me_mu_grad_beta!,
       # Variance computation primitives
       delta_method_se,
       accumulate_ame_gradient!,
       # Batch operations
       marginal_effects_batch!
