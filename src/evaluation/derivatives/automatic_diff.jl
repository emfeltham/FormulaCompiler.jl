# automatic_diff.jl
# ForwardDiff automatic differentiation implementations

# =============================================================================
# Custom Jacobian extraction — avoids ForwardDiff's reshape-based extraction
# =============================================================================
#
# ForwardDiff.extract_jacobian! uses reshape(result, ...) which allocates a
# 48-byte ReshapedArray wrapper when running under --check-bounds=yes (as
# Pkg.test() does). These functions replace that with direct loop indexing,
# achieving zero allocations regardless of bounds-checking mode.

"""
    _extract_jacobian_direct!(T, J, ydual, N)

Extract Jacobian partials from dual output vector into matrix J.
Zero-allocation replacement for ForwardDiff.extract_jacobian! that avoids reshape.
"""
@inline function _extract_jacobian_direct!(::Type{T}, J::Matrix{Float64},
                                           ydual::AbstractVector, N::Int) where {T}
    M = length(ydual)
    @inbounds for j in 1:M
        d = ydual[j]
        for i in 1:N
            J[j, i] = ForwardDiff.partials(T, d, i)
        end
    end
    return J
end

"""
    _extract_jacobian_chunk_direct!(T, J, ydual, offset, chunksize)

Extract Jacobian partials for a single chunk into columns offset+1:offset+chunksize.
Zero-allocation replacement for ForwardDiff.extract_jacobian_chunk!.
"""
@inline function _extract_jacobian_chunk_direct!(::Type{T}, J::Matrix{Float64},
                                                  ydual::AbstractVector,
                                                  offset::Int, chunksize::Int) where {T}
    M = length(ydual)
    @inbounds for j in 1:M
        d = ydual[j]
        for i in 1:chunksize
            J[j, offset + i] = ForwardDiff.partials(T, d, i)
        end
    end
    return J
end

"""
    fc_jacobian!(J, g, x, cfg)

Custom Jacobian computation that uses ForwardDiff's seeding and dual evaluation
but replaces the extraction step with direct loop indexing.

Achieves zero allocations even under --check-bounds=yes by avoiding the
reshape() call in ForwardDiff.extract_jacobian!.
"""
function fc_jacobian!(J::Matrix{Float64}, g::F, x::Vector{Float64},
                      cfg::ForwardDiff.JacobianConfig{T,V,N}) where {F,T,V,N}
    xlen = length(x)
    xdual = cfg.duals

    if N == xlen
        # Vector mode: single pass evaluates all partials at once
        ForwardDiff.seed!(xdual, x, cfg.seeds)
        ydual = g(xdual)
        _extract_jacobian_direct!(T, J, ydual, N)
    else
        # Chunk mode: multiple passes, N variables at a time
        seeds = cfg.seeds
        ForwardDiff.seed!(xdual, x)

        # Loop bounds
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1

        # First chunk
        ForwardDiff.seed!(xdual, x, 1, seeds)
        ydual = g(xdual)
        _extract_jacobian_chunk_direct!(T, J, ydual, 0, N)
        ForwardDiff.seed!(xdual, x, 1)

        # Middle chunks
        for c in 2:div(xlen - lastchunksize, N)
            i = (c - 1) * N + 1
            ForwardDiff.seed!(xdual, x, i, seeds)
            ydual = g(xdual)
            _extract_jacobian_chunk_direct!(T, J, ydual, i - 1, N)
            ForwardDiff.seed!(xdual, x, i)
        end

        # Final chunk
        ForwardDiff.seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        ydual = g(xdual)
        _extract_jacobian_chunk_direct!(T, J, ydual, lastchunkindex - 1, lastchunksize)
    end

    return J
end

"""
    derivative_modelrow!(J, de::ADEvaluator, row) -> J

Primary automatic differentiation API — zero allocations via ForwardDiff dual arithmetic.

Computes the Jacobian of the compiled formula evaluated at the specified data row.

# Arguments
- `J::AbstractMatrix{Float64}`: Preallocated Jacobian buffer of size `(n_terms, n_vars)`
- `de::ADEvaluator`: AD evaluator built by `derivativeevaluator(:ad, compiled, data, vars)`
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `J`: The same matrix passed in, now containing `J[i,j] = ∂X[i]/∂vars[j]` for the specified row

# Performance Characteristics
- Memory: 0 bytes allocated (even under --check-bounds=yes)
- Speed: ~30-50ns for simple formulas, ~1-2μs for complex formulas
- Accuracy: Machine precision derivatives via ForwardDiff dual arithmetic

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

    # Custom Jacobian: uses ForwardDiff seeding/duals but avoids reshape allocation
    fc_jacobian!(J, ctx.g, input_vec, ctx.cfg)

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
