# finite_diff.jl
# Finite difference implementations

# Mathematically appropriate auto step scale for central differences
# h ≈ cbrt(eps(Float64)) * max(1, |x|)
const FD_AUTO_EPS_SCALE = cbrt(eps(Float64))  # ≈ 6.055454452393339e-6

"""
    derivative_modelrow_fd!(J, compiled, data, row; vars, step=:auto) -> J

Compute Jacobian matrix using finite differences with central difference approximation (standalone version).

Standalone finite difference implementation that builds temporary override structures
for each call. Provides robust numerical differentiation with adaptive step sizing,
suitable for validation and environments where pre-built evaluators are not available.

# Arguments
- `J::AbstractMatrix{Float64}`: Preallocated Jacobian buffer of size `(n_terms, n_vars)`
  - Will be overwritten with partial derivatives ∂X[i]/∂vars[j]
- `compiled::UnifiedCompiled`: Compiled formula from `compile_formula(model, data)`
- `data::NamedTuple`: Data in column-table format (from `Tables.columntable(df)`)
- `row::Int`: Row index to evaluate (1-based indexing)
- `vars::Vector{Symbol}`: Continuous variables to differentiate with respect to
- `step`: Step size for finite differences
  - `:auto`: Adaptive sizing `h = ε^(1/3) * max(1, |x|)` where ε = machine epsilon
  - `Float64`: Fixed step size for all variables

# Returns
- `J`: The same matrix passed in, containing `J[i,j] = ∂X[i]/∂vars[j]` via finite differences

# Performance
- **Memory**: Allocates temporary override structures per call
- **Computation**: Two model evaluations per variable (central differences)
- **Accuracy**: Good numerical accuracy with mathematically appropriate step sizes
- **Alternative**: For zero allocations, use `derivative_modelrow_fd_pos!` with pre-built evaluator

# Mathematical Method
Uses central difference approximation:
```
∂f/∂x ≈ [f(x + h) - f(x - h)] / (2h)
```
with adaptive step sizing for numerical stability.

# Example
```julia
using FormulaCompiler, GLM

# Setup model
model = lm(@formula(y ~ x * group + log(abs(z) + 1)), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Standalone finite differences
vars = [:x, :z]
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow_fd!(J, compiled, data, 1; vars=vars)

# Adaptive step sizing (recommended)
derivative_modelrow_fd!(J, compiled, data, 1; vars=vars, step=:auto)

# Fixed step size
derivative_modelrow_fd!(J, compiled, data, 1; vars=vars, step=1e-6)
```

# Use Cases
- **Validation**: Cross-check automatic differentiation results
- **Fallback computation**: When ForwardDiff is unavailable or problematic
- **Numerical verification**: Validate derivative implementations
- **Educational purposes**: Understand finite difference mechanics

# Step Size Selection
- **`:auto`**: Recommended for most applications, balances truncation and roundoff error
- **Fixed values**: Use when specific step size control is needed
- **Variable-specific**: Each variable gets step proportional to its magnitude

See also: [`derivative_modelrow_fd_pos!`](@ref) for zero-allocation version, [`derivative_modelrow!`](@ref) for AD
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
    # Set row for overrides (unrolled) BEFORE reading base values
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    # Fill xbase from the underlying base vectors to avoid reading a stale replacement
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.overrides[$j].base[row]))
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
        push!(stmts, :(h = (FD_AUTO_EPS_SCALE * max(1.0, abs(x)))))
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
    # Set row for overrides BEFORE reading base values
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    # Fill xbase from underlying base vectors
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.overrides[$j].base[row]))
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
            # Step selection (auto uses cbrt(eps(Float64)))
            h = step === :auto ? (FD_AUTO_EPS_SCALE * max(1.0, abs(x))) : step
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

"""
    fd_jacobian_column!(Jk, de, row, var; step=:auto)

Fill `Jk` with the k-th Jacobian column ∂X/∂var at the given row using finite differences.

Arguments:
- `Jk::Vector{Float64}`: Preallocated buffer of length `n_terms`
- `de::DerivativeEvaluator`: Built by `build_derivative_evaluator`
- `row::Int`: Row index (1-based)
- `var::Symbol`: Variable to differentiate with respect to (must be in `de.vars`)
- `step`: Numeric step size or `:auto` (`eps()^(1/3) * max(1, |x|)`)

Returns:
- The same `Jk` buffer, with `Jk[i] = ∂X[i]/∂var` for the given row

Notes:
- Zero allocations per call after warmup
- Uses typed overrides and cached column access for performance
- Variable must exist in `de.vars` from evaluator construction
"""
function fd_jacobian_column!(
    Jk::Vector{Float64},
    de::DerivativeEvaluator,
    row::Int,
    var::Symbol;
    step=:auto,
)
    # Find variable index
    var_idx = findfirst(==(var), de.vars)
    var_idx === nothing && throw(ArgumentError("Variable $var not found in de.vars"))
    
    @assert length(Jk) == length(de)
    
    if step === :auto
        return _fd_column_auto!(Jk, de, row, var_idx)
    else
        return _fd_column_step!(Jk, de, row, var_idx, Float64(step))
    end
end

# Internal FD single column (no keyword) with auto step
@generated function _fd_column_auto!(
    Jk::Vector{Float64},
    de::DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC},
    row::Int,
    var_idx::Int,
) where {T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}
    N = NV
    stmts = Expr[]
    push!(stmts, :(yplus = de.fd_yplus))
    push!(stmts, :(yminus = de.fd_yminus))
    push!(stmts, :(xbase = de.fd_xbase))
    push!(stmts, :(nterms = length(de)))
    
    # Set row for overrides BEFORE reading base values
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    
    # Fill xbase from underlying base vectors (avoid stale replacement)
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.overrides[$j].base[row]))
    end
    
    # Single variable finite difference computation
    push!(stmts, :(x = xbase[var_idx]))
    
    # Set all overrides to base values
    for k in 1:N
        push!(stmts, :(@inbounds de.overrides[$k].replacement = xbase[$k]))
    end
    
    # Step selection and evaluations for the single variable
    push!(stmts, :(h = (FD_AUTO_EPS_SCALE * max(1.0, abs(x)))))
    push!(stmts, :(@inbounds de.overrides[var_idx].replacement = x + h))
    push!(stmts, :(de.compiled_base(yplus, de.data_over, row)))
    push!(stmts, :(@inbounds de.overrides[var_idx].replacement = x - h))
    push!(stmts, :(de.compiled_base(yminus, de.data_over, row)))
    
    # Central difference for single column
    push!(stmts, :(inv_2h = 1.0 / (2.0 * h)))
    push!(stmts, quote
        @fastmath for i in 1:nterms
            @inbounds Jk[i] = (yplus[i] - yminus[i]) * inv_2h
        end
    end)
    
    return Expr(:block, stmts...)
end

# Internal FD single column (no keyword) with explicit step  
@generated function _fd_column_step!(
    Jk::Vector{Float64},
    de::DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC},
    row::Int,
    var_idx::Int,
    step::Float64,
) where {T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}
    N = NV
    stmts = Expr[]
    push!(stmts, :(yplus = de.fd_yplus))
    push!(stmts, :(yminus = de.fd_yminus))
    push!(stmts, :(xbase = de.fd_xbase))
    push!(stmts, :(nterms = length(de)))
    
    # Set row for overrides BEFORE reading base values
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    
    # Fill xbase from underlying base vectors
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.overrides[$j].base[row]))
    end
    
    push!(stmts, :(x = xbase[var_idx]))
    
    for k in 1:N
        push!(stmts, :(@inbounds de.overrides[$k].replacement = xbase[$k]))
    end
    
    push!(stmts, :(h = step))
    push!(stmts, :(@inbounds de.overrides[var_idx].replacement = x + h))
    push!(stmts, :(de.compiled_base(yplus, de.data_over, row)))
    push!(stmts, :(@inbounds de.overrides[var_idx].replacement = x - h))
    push!(stmts, :(de.compiled_base(yminus, de.data_over, row)))
    
    push!(stmts, :(inv_2h = 1.0 / (2.0 * h)))
    push!(stmts, quote
        @fastmath for i in 1:nterms
            @inbounds Jk[i] = (yplus[i] - yminus[i]) * inv_2h
        end
    end)
    
    return Expr(:block, stmts...)
end

"""
    fd_jacobian_column_pos!(Jk, de, row, var_idx)

Positional hot path for single-column finite-difference Jacobian.
Uses variable index directly to avoid symbol lookup.
"""
@inline fd_jacobian_column_pos!(Jk::Vector{Float64}, de::DerivativeEvaluator, row::Int, var_idx::Int) = _fd_column_auto!(Jk, de, row, var_idx)
