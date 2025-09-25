# derivatives.jl - Main interface for derivative computations
#
# High-level public API and exports for the derivative system.
# The implementation is organized into focused modules in the derivatives/ subdirectory.


# Load all derivative system modules
include("derivatives/overrides.jl")
include("derivatives/types.jl") 
include("derivatives/evaluator.jl")
include("derivatives/automatic_diff.jl")
include("derivatives/finite_diff.jl")
include("derivatives/marginal_effects.jl")
include("derivatives/contrasts.jl")
include("derivatives/link_functions.jl")
include("derivatives/batch.jl")
include("derivatives/utilities.jl")

# Export public API
export build_derivative_evaluator,
       derivative_modelrow!, derivative_modelrow,
       derivative_modelrow_fd!, derivative_modelrow_fd, derivative_modelrow_fd_pos!,
       # High-level dispatcher functions with backend selection
       marginal_effects_eta!, marginal_effects_mu!,
       # High-performance direct function calls
       marginal_effects_eta_ad!, marginal_effects_eta_fd!,
       marginal_effects_mu_ad!, marginal_effects_mu_fd!,
       marginal_effects_eta_grad!, marginal_effects_eta_grad,
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
