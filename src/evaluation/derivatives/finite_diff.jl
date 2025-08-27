# finite_diff.jl - Finite difference implementations

"""
    derivative_modelrow_fd!(J, compiled, data, row; vars, step=:auto)

Finite-difference Jacobian for a single row using central differences (standalone).

Arguments:
- `J::AbstractMatrix{Float64}`: Preallocated `(n_terms, n_vars)` buffer.
- `compiled::UnifiedCompiled`: Result of `compile_formula`.
- `data::NamedTuple`: Column-table data.
- `row::Int`: Row index.
- `vars::Vector{Symbol}`: Variables to differentiate with respect to.
- `step`: Numeric step size or `:auto` (`eps()^(1/3) * max(1, |x|)`).

Notes:
- Two evaluations per variable; useful as a robust fallback and for cross-checks.
- This standalone path allocates per call (builds per-call overrides and small temporaries). For zero allocations after warmup, prefer the evaluator FD path (`derivative_modelrow_fd_pos!`).
"""
function derivative_modelrow_fd!(
    J::AbstractMatrix{Float64},
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    row::Int;
    vars::Vector{Symbol},
    step=:auto,
) where {T, Ops, S, O}
    @assert size(J, 1) == length(compiled)
    @assert size(J, 2) == length(vars)
    # Build row-local override once for all vars
    # Use untyped overrides for maximal compatibility/correctness here
    data_over, overrides = build_row_override_data(data, vars, row)
    # Buffers
    yplus = Vector{Float64}(undef, length(compiled))
    yminus = Vector{Float64}(undef, length(compiled))
    # Base values for each var at row
    xbase = similar(yplus, length(vars))
    for (j, s) in enumerate(vars)
        xbase[j] = getproperty(data, s)[row]
    end
    # Iterate variables
    for (j, s) in enumerate(vars)
        x = xbase[j]
        # Set all other overrides to base
        for (k, sk) in enumerate(vars)
            overrides[k].replacement = xbase[k]
        end
        # Step selection
        h = step === :auto ? (eps(Float64)^(1/3) * max(1.0, abs(float(x)))) : step
        # Plus
        overrides[j].replacement = x + h
        compiled(yplus, data_over, row)
        # Minus
        overrides[j].replacement = x - h
        compiled(yminus, data_over, row)
        # Central difference column
        @inbounds @fastmath for i in 1:length(compiled)
            J[i, j] = (yplus[i] - yminus[i]) / (2h)
        end
    end
    return J
end

"""
    derivative_modelrow_fd!(J, evaluator, row; step=:auto)

Finite-difference Jacobian via an evaluator with preallocated state. This path is optimized
for zero allocations after warmup using typed overrides and unrolled column access.

Notes:
- For guaranteed zero allocations, use the positional hot path `derivative_modelrow_fd_pos!(J, de, row)`.
- The keyword-based wrapper forwards to allocation-free internals but may register tiny
  environment-dependent overhead in some benchmarks; the positional variant avoids this.
"""
# Internal FD evaluator (no keyword) with auto step
@generated function _derivative_modelrow_fd_auto!(
    J::AbstractMatrix{Float64},
    de::DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC},
    row::Int,
) where {T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}
    N = NV
    stmts = Expr[]
    push!(stmts, :(yplus = de.fd_yplus))
    push!(stmts, :(yminus = de.fd_yminus))
    push!(stmts, :(xbase = de.fd_xbase))
    push!(stmts, :(nterms = length(de)))
    # Fill xbase with unrolled tuple access
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.fd_columns[$j][row]))
    end
    # Set row for overrides (unrolled)
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    # Main unrolled finite difference loop across variables
    for j in 1:N
        # x = xbase[j]
        push!(stmts, :(x = xbase[$j]))
        # set overrides[k].replacement = xbase[k] for all k
        for k in 1:N
            push!(stmts, :(@inbounds de.overrides[$k].replacement = xbase[$k]))
        end
        # step selection and evaluations
        push!(stmts, :(h = (2.220446049250313e-6 * max(1.0, abs(x)))))
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x + h))
        push!(stmts, :(de.compiled_base(yplus, de.data_over, row)))
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x - h))
        push!(stmts, :(de.compiled_base(yminus, de.data_over, row)))
        push!(stmts, :(inv_2h = 1.0 / (2.0 * h)))
        push!(stmts, quote
            @fastmath for i in 1:nterms
                @inbounds J[i, $j] = (yplus[i] - yminus[i]) * inv_2h
            end
        end)
    end
    return Expr(:block, stmts...)
end

# Internal FD evaluator (no keyword) with explicit step
@generated function _derivative_modelrow_fd_step!(
    J::AbstractMatrix{Float64},
    de::DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC},
    row::Int,
    step::Float64,
) where {T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}
    N = NV
    stmts = Expr[]
    push!(stmts, :(yplus = de.fd_yplus))
    push!(stmts, :(yminus = de.fd_yminus))
    push!(stmts, :(xbase = de.fd_xbase))
    push!(stmts, :(nterms = length(de)))
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.fd_columns[$j][row]))
    end
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    for j in 1:N
        push!(stmts, :(x = xbase[$j]))
        for k in 1:N
            push!(stmts, :(@inbounds de.overrides[$k].replacement = xbase[$k]))
        end
        push!(stmts, :(h = step))
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x + h))
        push!(stmts, :(de.compiled_base(yplus, de.data_over, row)))
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x - h))
        push!(stmts, :(de.compiled_base(yminus, de.data_over, row)))
        push!(stmts, :(inv_2h = 1.0 / (2.0 * h)))
        push!(stmts, quote
            @fastmath for i in 1:nterms
                @inbounds J[i, $j] = (yplus[i] - yminus[i]) * inv_2h
            end
        end)
    end
    return Expr(:block, stmts...)
end

# Public FD evaluator with keyword dispatch forwarding to allocation-free internals
@inline function derivative_modelrow_fd!(
    J::AbstractMatrix{Float64},
    de::DerivativeEvaluator,
    row::Int; step=:auto,
)
    if step === :auto
        return _derivative_modelrow_fd_auto!(J, de, row)
    else
        return _derivative_modelrow_fd_step!(J, de, row, Float64(step))
    end
end

"""
    derivative_modelrow_fd_pos!(J, evaluator, row)

Positional hot path for finite-difference Jacobian via an evaluator.
Writes into `J` and performs zero allocations per call after warmup.

Notes:
- Uses preallocated evaluator state (typed overrides, unrolled column access).
- Prefer this for production/bulk evaluation. For a standalone baseline, see
  `derivative_modelrow_fd!(J, compiled, data, row; vars)` (allocates by design).
"""
@inline derivative_modelrow_fd_pos!(J::AbstractMatrix{Float64}, de::DerivativeEvaluator, row::Int) = _derivative_modelrow_fd_auto!(J, de, row)

function derivative_modelrow_fd(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    row::Int;
    vars::Vector{Symbol},
    step=:auto,
) where {T, Ops, S, O}
    J = Matrix{Float64}(undef, length(compiled), length(vars))
    derivative_modelrow_fd!(J, compiled, data, row; vars=vars, step=step)
    return J
end

"""
    marginal_effects_eta_fd!(g, evaluator, beta, row; step=:auto)

Zero-allocation marginal effects using finite differences (with override system).
"""
@generated function marginal_effects_eta_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    step=:auto,
)
    quote
        # Direct access to preallocated buffers
        yplus = de.fd_yplus
        yminus = de.fd_yminus
        xbase = de.fd_xbase
        J = de.jacobian_buffer
        
        nvars = length(de.vars)
        nterms = length(de)
        
        # Get base values using pre-cached columns (no getproperty)
        @inbounds for j in 1:nvars
            xbase[j] = de.fd_columns[j][row]
        end
        
        # Set row for overrides
        @inbounds for i in 1:nvars
            de.overrides[i].row = row
        end
        
        # Main finite difference loop - compile-time optimized
        @inbounds for j in 1:nvars
            x = xbase[j]
            # Set all overrides to base values
            for k in 1:nvars
                de.overrides[k].replacement = xbase[k]
            end
            # Step selection (hardcode eps()^(1/3) to avoid function calls)
            h = step === :auto ? (2.220446049250313e-6 * max(1.0, abs(x))) : step
            # Plus
            de.overrides[j].replacement = x + h
            de.compiled_base(yplus, de.data_over, row)
            # Minus  
            de.overrides[j].replacement = x - h
            de.compiled_base(yminus, de.data_over, row)
            # Store Jacobian column with FMA optimization
            inv_2h = 1.0 / (2.0 * h)
            @fastmath for i in 1:nterms
                J[i, j] = (yplus[i] - yminus[i]) * inv_2h
            end
        end
        
        # Matrix multiplication: g = J' * β (fully optimized with FMA)
        @inbounds @fastmath for j in 1:nvars
            sum_val = 0.0
            for i in 1:nterms
                sum_val = muladd(J[i, j], beta[i], sum_val)
            end
            g[j] = sum_val
        end
        
        return g
    end
end

function marginal_effects_eta_fd(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    step=:auto,
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta_fd!(g, de, beta, row; step=step)
    return g
end