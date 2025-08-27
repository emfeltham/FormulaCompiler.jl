# derivatives.jl - Main interface for derivative computations
#
# High-level public API and exports for the derivative system.
# The implementation is organized into focused modules in the derivatives/ subdirectory.

# TODO(derivatives): See DERIVATIVE_PLAN.md → Typing Checklist & Acceptance Criteria.
# - Eliminate Any on hot paths (closure/config/dual fields, overrides, column cache).
# - Prefer gradient-based η path for strict zero-allocation marginal effects.
# - Keep FD Jacobian as 0-alloc fallback; AD Jacobian 0-alloc may be env-dependent.
#
# Contributor TODOs (typing and allocation hygiene):
# 1) Make evaluator fields concrete
#    - g::Any             → store concrete closure type (DerivClosure{DE})
#    - cfg::Any           → store concrete ForwardDiff.JacobianConfig{…}
#    - rowvec_dual::Any   → Vector{<:ForwardDiff.Dual{…}} with concrete eltype
#    - compiled_dual::Any → UnifiedCompiled{<:ForwardDiff.Dual{…},Ops,S,O}
# 2) Column cache
#    - fd_columns::Vector{Any} → Vector{<:AbstractVector{T}} or NTuple for fixed nvars
# 3) Overrides
#    - SingleRowOverrideVector <: AbstractVector{Any}
#      replace with SingleRowOverrideVector{T} and build per-eltype data_over (Float64 & Dual)
# 4) Config creation
#    - Ensure JacobianConfig/GradientConfig are built once with concrete closure
# 5) Tests
#    - Tighten FD Jacobian to 0 allocations; η-gradient to 0; gate AD Jacobian per env caps

# Load all derivative system modules
include("derivatives/overrides.jl")
include("derivatives/types.jl") 
include("derivatives/evaluator.jl")
include("derivatives/automatic_diff.jl")
include("derivatives/finite_diff.jl")
include("derivatives/marginal_effects.jl")
include("derivatives/contrasts.jl")
include("derivatives/link_functions.jl")
include("derivatives/utilities.jl")

# Export public API
export build_derivative_evaluator,
       derivative_modelrow!, derivative_modelrow,
       derivative_modelrow_fd!, derivative_modelrow_fd, derivative_modelrow_fd_pos!,
       marginal_effects_eta!, marginal_effects_eta,
       marginal_effects_mu!, marginal_effects_mu,
       marginal_effects_eta_grad!, marginal_effects_eta_grad,
       contrast_modelrow!, contrast_modelrow,
       continuous_variables,
       # Single-column FD and parameter gradient functions
       fd_jacobian_column!, fd_jacobian_column_pos!,
       me_eta_grad_beta!,
       me_mu_grad_beta!,
       # Variance computation primitives
       delta_method_se,
       accumulate_ame_gradient!