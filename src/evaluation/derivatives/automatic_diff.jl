# automatic_diff.jl
# ForwardDiff automatic differentiation implementations

"""
    derivative_modelrow!(J, evaluator, row) -> J

Compute Jacobian matrix of model row with respect to continuous variables using automatic differentiation.

Fills a preallocated Jacobian matrix with partial derivatives ∂X[i]/∂vars[j] where X is the
model matrix row and vars are the differentiation variables. Uses ForwardDiff.jl for accurate,
efficient automatic differentiation with dual numbers.

# Arguments
- `J::AbstractMatrix{Float64}`: Preallocated Jacobian buffer of size `(n_terms, n_vars)`
  - `n_terms = length(compiled)` (number of model matrix columns)
  - `n_vars = length(evaluator.vars)` (number of differentiation variables)
  - Contents will be overwritten with partial derivatives
- `evaluator::DerivativeEvaluator`: Built by `build_derivative_evaluator(compiled, data; vars=...)`
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `J`: The same matrix passed in, now containing `J[i,j] = ∂X[i]/∂vars[j]` for the specified row

# Performance
- **Memory**: Small allocations per call (ForwardDiff internals, unavoidable)
- **Accuracy**: Machine precision derivatives via dual numbers
- **Alternative**: For zero allocations, use `derivative_modelrow_fd!()`

# Mathematical Details
Computes the Jacobian matrix:
```
J[i,j] = ∂(model_matrix_row[i])/∂(vars[j])
```
where the derivatives account for all formula transformations including interactions,
functions, and categorical contrasts.

# Example
```julia
using FormulaCompiler, GLM

# Setup model with interactions and functions
model = lm(@formula(y ~ x * group + log(abs(z) + 1)), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build evaluator for continuous variables
vars = [:x, :z]
de = build_derivative_evaluator(compiled, data, vars)

# Compute Jacobian for row 1
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de, 1)

# J[1,1] = ∂(intercept)/∂x = 0
# J[2,1] = ∂x/∂x = 1  
# J[3,1] = ∂(group_B)/∂x = 0 (categorical terms)
# J[4,1] = ∂(x*group_B)/∂x = group_B_value (interaction derivative)
# J[5,1] = ∂(log(|z|+1))/∂x = 0
# J[2,2] = ∂x/∂z = 0
# J[5,2] = ∂(log(|z|+1))/∂z = sign(z)/(|z|+1) (chain rule)
```

# Use Cases
- **Marginal effects**: Economic impact analysis
- **Sensitivity analysis**: Model robustness assessment
- **Gradient computation**: Custom optimization and inference
- **Uncertainty propagation**: Delta method standard errors

See also: [`derivative_modelrow_fd!`](@ref) for finite differences alternative
"""
@inline function _dualtype_for_vars(nvars::Int)
    return ForwardDiff.Dual{Nothing, Float64, nvars}
end

@inline function _seed_duals_identity!(xdual::Vector{T}, partials_unit_vec, data, vars::Vector{Symbol}, row::Int) where {Tag, N, T<:ForwardDiff.Dual{Tag, Float64, N}}
    @inbounds for i in 1:length(vars)
        v0 = getproperty(data, vars[i])[row]
        v = Float64(v0)
        xdual[i] = T(v, partials_unit_vec[i])
    end
    return xdual
end

function derivative_modelrow!(J::AbstractMatrix{Float64}, de::DerivativeEvaluator, row::Int)
    @assert size(J, 1) == length(de) "Jacobian row mismatch: expected $(length(de)) terms"
    @assert size(J, 2) == length(de.vars) "Jacobian column mismatch: expected $(length(de.vars)) variables"
    # Manual dual-evaluation path (zero-alloc after warmup)
    de.row = row
    # Typed single-cache path (Tag = Nothing, N = length(vars))
    N = length(de.vars)
    _seed_duals_identity!(de.x_dual_vec, de.partials_unit_vec, de.base_data, de.vars, row)
    @inbounds for i in 1:N
        ov = de.overrides_dual_vec[i]
        ov.row = row
        ov.replacement = de.x_dual_vec[i]
    end
    de.compiled_dual_vec(de.rowvec_dual_vec, de.data_over_dual_vec, row)
    @inbounds for i in 1:size(J,1)
        di = de.rowvec_dual_vec[i]
        parts = ForwardDiff.partials(di)
        for j in 1:N
            J[i,j] = parts[j]
        end
    end
    return J
end

"""
    derivative_modelrow(deval, row) -> Matrix{Float64}

Allocating convenience wrapper that returns the Jacobian for one row.
"""
function derivative_modelrow(de::DerivativeEvaluator, row::Int)
    J = Matrix{Float64}(undef, length(de), length(de.vars))
    derivative_modelrow!(J, de, row)
    return J
end

"""
    marginal_effects_eta_grad!(g, de, beta, row)

Compute marginal effects on η = Xβ via ForwardDiff.gradient! of the scalar
function h(x) = dot(β, Xrow(x)). Uses the existing AD closure to evaluate rows.
"""
function marginal_effects_eta_grad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
)
    @assert length(g) == length(de.vars)
    @assert length(beta) == length(de)
    # Manual dual-evaluation path (compute J, then g = J' * β)
    de.row = row
    # Beta handling without allocations
    if beta isa Vector{Float64}
        de.beta_ref[] = beta
    else
        @assert length(de.beta_buf) == length(beta)
        copyto!(de.beta_buf, beta)
        de.beta_ref[] = de.beta_buf
    end
    # Build Jacobian into the evaluator's preallocated buffer
    derivative_modelrow!(de.jacobian_buffer, de, row)
    # Compute g = J' * β without allocations
    @inbounds begin
        nterms = size(de.jacobian_buffer, 1)
        nvars = size(de.jacobian_buffer, 2)
        @assert nvars == length(de.vars)
        βref = de.beta_ref[]
        for j in 1:nvars
            acc = 0.0
            for i in 1:nterms
                acc += de.jacobian_buffer[i, j] * βref[i]
            end
            g[j] = acc
        end
    end
    return g
end

function marginal_effects_eta_grad(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta_grad!(g, de, beta, row)
    return g
end

