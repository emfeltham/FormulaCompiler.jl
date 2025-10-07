# gradients.jl - Parameter Gradient Utilities

"""
    delta_method_se(gβ, Σ)

Compute standard error using delta method: SE = sqrt(gβ' * Σ * gβ)

Arguments:
- `gβ::AbstractVector{Float64}`: Parameter gradient vector
- `Σ::AbstractMatrix{Float64}`: Parameter covariance matrix from model

Returns:
- `Float64`: Standard error

Notes:
- Zero allocations per call
- Implements Var(m) = gβ' Σ gβ where m is marginal effect
- Works with gradients computed by any backend (AD, FD, analytical)
"""
function delta_method_se(gβ::AbstractVector{Float64}, Σ::AbstractMatrix{Float64})
    # Zero-allocation computation of sqrt(gβ' * Σ * gβ)
    # Use BLAS dot product to avoid temporary arrays
    n = length(gβ)
    result = 0.0
    @inbounds for i in 1:n
        temp = 0.0
        for j in 1:n
            temp += Σ[i, j] * gβ[j]
        end
        result += gβ[i] * temp
    end

    # Debug check for negative variance (should not happen with valid covariance matrix)
    if result < 0.0
        @warn "Negative variance detected in delta method: gβ'Σgβ = $result. " *
              "This suggests numerical issues or invalid covariance matrix. " *
              "Check gradient computation and covariance matrix conditioning."
        return NaN
    end

    return sqrt(result)
end

# =============================================================================
# DEPRECATION NOTICE (2025-10-07)
# =============================================================================
# The delta_method_se function has been migrated to Margins.jl as part of the
# separation between computational primitives (FormulaCompiler) and statistical
# interface (Margins).
#
# This function will be REMOVED in FormulaCompiler v2.0
#
# Migration guide:
#   Before (FormulaCompiler v1.x):
#     using FormulaCompiler
#     se = delta_method_se(gβ, Σ)
#
#   After (FormulaCompiler v1.1+ with Margins):
#     using FormulaCompiler  # For primitives
#     using Margins          # For statistical interface
#     se = delta_method_se(gβ, Σ)  # Now from Margins.jl
#
# The function maintains the same API and performance characteristics.
# =============================================================================
