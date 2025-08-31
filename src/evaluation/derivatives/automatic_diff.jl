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
de = build_derivative_evaluator(compiled, data; vars=vars)

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

See also: [`derivative_modelrow_fd!`](@ref) for zero-allocation alternative, [`marginal_effects_eta!`](@ref)
"""
function derivative_modelrow!(J::AbstractMatrix{Float64}, de::DerivativeEvaluator, row::Int)
    @assert size(J, 1) == length(de) "Jacobian row mismatch: expected $(length(de)) terms"
    @assert size(J, 2) == length(de.vars) "Jacobian column mismatch: expected $(length(de.vars)) variables"
    # Set row and seed x with base values
    de.row = row
    for (i, s) in enumerate(de.vars)
        de.xbuf[i] = getproperty(de.base_data, s)[row]
    end
    ForwardDiff.jacobian!(J, de.g, de.xbuf, de.cfg)
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
    # Seed x with base values
    de.row = row
    for (i, s) in enumerate(de.vars)
        de.xbuf[i] = getproperty(de.base_data, s)[row]
    end
    # Use stored scalar closure/config, update beta reference
    de.beta_ref[] = (beta isa Vector{Float64} ? beta : Vector{Float64}(beta))
    ForwardDiff.gradient!(g, de.gscalar, de.xbuf, de.gradcfg)
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