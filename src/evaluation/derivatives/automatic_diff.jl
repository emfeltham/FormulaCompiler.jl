# automatic_diff.jl
# ForwardDiff automatic differentiation implementations

"""
    derivative_modelrow!(J, de::ADEvaluator, row) -> J

Primary automatic differentiation API - zero allocations via ForwardDiff.jacobian!.

Phase 2 implementation using cached ForwardDiff configuration for zero allocations.
Replaces manual dual construction with ForwardDiff's optimized jacobian! routine.

# Arguments
- `J::AbstractMatrix{Float64}`: Preallocated Jacobian buffer of size `(n_terms, n_vars)`
- `de::ADEvaluator`: AD evaluator built by `derivativeevaluator(:ad, compiled, data, vars)`
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `J`: The same matrix passed in, now containing `J[i,j] = ∂X[i]/∂vars[j]` for the specified row

# Performance Characteristics (Phase 2)
- **Memory**: 0 bytes allocated (cached buffers and ForwardDiff config)
- **Speed**: Target ~60ns with ForwardDiff.jacobian! optimization
- **Accuracy**: Machine precision derivatives via ForwardDiff dual arithmetic

# Example
```julia
using FormulaCompiler, GLM

# Setup model
model = lm(@formula(y ~ x + z), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build AD evaluator
de = derivativeevaluator(:ad, compiled, data, [:x, :z])

# Zero-allocation Jacobian computation
J = Matrix{Float64}(undef, length(compiled), length(de.vars))
derivative_modelrow!(J, de, 1)  # 0 bytes allocated
```
"""
function derivative_modelrow!(J::AbstractMatrix{Float64}, de::ADEvaluator, row::Int)
    ctx = de.ctx
    core = de.core

    @assert size(J, 1) == length(de) "Jacobian row mismatch: expected $(length(de)) terms"
    @assert size(J, 2) == length(de.vars) "Jacobian column mismatch: expected $(length(de.vars)) variables"

    # Update evaluator row reference (used by closure)
    set_row!(core, row)

    # Use preallocated input buffer from JacobianContext
    input_vec = ctx.input_vec
    columns = ctx.var_columns
    @inbounds for i in eachindex(columns)
        input_vec[i] = Float64(columns[i][row])
    end

    # Call ForwardDiff.jacobian! with cached config (zero allocations)
    ForwardDiff.jacobian!(J, ctx.g, input_vec, ctx.cfg)

    return J
end

# =============================================================================
# REMOVED (2025-10-07): marginal_effects_eta! migrated to Margins.jl v2.0
# =============================================================================
# The marginal_effects_eta! statistical interface functions have been migrated
# to Margins.jl. FormulaCompiler retains the computational primitives:
# - derivative_modelrow! (Jacobian computation)  
# - _matrix_multiply_eta! (internal utility)
#
# To use marginal effects:
#   using Margins
#   marginal_effects_eta!(g, de, β, row)
# =============================================================================
