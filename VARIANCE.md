# Variance and Standard Errors for Marginal Effects

This note explains how to compute standard errors (SEs) for marginal effects using the delta method in a way that remains zero‑allocation per call with the current evaluator design. The key point: you do not need the full Jacobian ∂X/∂x to get SEs — you only need per‑effect gradients with respect to the model parameters β.

## Delta Method (what you actually need)

- Let m be a scalar marginal effect (for one variable, at one row or averaged). The delta method gives
  
  Var(m) = gβ' Σ gβ,   where gβ = ∂m/∂β and Σ = Var(β) from the fitted model.

- Standard error: SE(m) = sqrt(gβ' Σ gβ).

Therefore, you need gβ for each marginal effect m. You do not need the full ∂X/∂x for all variables at once.

## Per‑row formulas (η and μ cases)

Let X_row be the model row (length p), η = X_row' β, and J_k = ∂X/∂x_k (the k‑th column of the Jacobian ∂X/∂x). All vectors are length p (number of terms).

- Marginal effect on η with respect to x_k:
  - m_k = J_k' β
  - Gradient wrt β: gβ = ∂m_k/∂β = J_k

- Marginal effect on μ with respect to x_k (GLM link g, so μ = g⁻¹(η)):
  - m_k = g'(η) (J_k' β)
  - Gradient wrt β: gβ = g'(η) J_k + (J_k' β) g''(η) X_row

Notes:
- g'(η), g''(η) are analytic link derivatives; we already provide fast inline implementations.
- J_k is the k‑th column of ∂X/∂x; it can be computed via the zero‑allocation FD evaluator (single variable perturbation) without building the full Jacobian.

## Zero‑allocation computation (per effect)

For one marginal effect with respect to variable x_k at row i:

1) Compute J_k (one column) using the FD evaluator (single‑column mode):
   - fd_jacobian_column!(Jk, de, row=i, var=:xk)

2) Compute X_row in place and η:
   - compiled(Xrow, data, i)
   - η = dot(β, Xrow)

3) Build gβ:
   - η case: gβ .= J_k
   - μ case: gβ .= g'(η) .* J_k .+ (dot(J_k, β) .* g''(η)) .* X_row

4) Standard error:
   - se = sqrt(gβ' Σ gβ)

All steps are in‑place and reuse preallocated buffers; no per‑call allocations are needed with the FD evaluator and compiled row path.

## Average Marginal Effects (AME)

- If AME is the average of per‑row marginal effects m_k(i), the gradient wrt β is the average of per‑row gradients (linearity):
  
  gβ(AME) = (1/n) Σ_i gβ(i)

- Algorithm:
  - Accumulate gβ(i) over rows into a running buffer, then divide by n.
  - Apply the delta method once using Σ from the fitted model: se = sqrt(gβ(AME)' Σ gβ(AME)).

## Practical API suggestions

- Single‑column FD Jacobian (avoid building full J):
  - `fd_jacobian_column!(Jk::Vector{Float64}, de, row::Int, var::Symbol)` → fills J_k

- Gradients wrt β (in‑place):
  - `me_eta_grad_beta!(gβ, de, β, row, var)` → gβ .= J_k
  - `me_mu_grad_beta!(gβ, de, β, row, var; link)` → gβ .= g'(η) J_k + (J_k' β) g''(η) X_row

- Standard errors via delta method:
  - `se = sqrt(gβ' * Σ * gβ)` where Σ comes from the fitted model (e.g., vcov).

## Why you don’t need the full ∂X/∂x

- SEs require gβ = ∂m/∂β, not ∂X/∂x in full. For each marginal effect, a single column J_k plus X_row and link derivatives suffice.
- Computing J_k on demand via the FD evaluator keeps memory and runtime predictable and allocation‑free after warmup.
- If you need the full J for other reasons, you can still build it; but SEs for marginal effects do not require it.

## Recommendations

- For production (large n_rows): use the FD evaluator single‑column mode to compute J_k and build gβ in place; then apply the delta method. This achieves strict zero‑allocation per call after the evaluator is built.
- For small problems or validation: AD Jacobian can be used as a cross‑check (accept small, environment‑dependent allocations).

## Complete Workflow Examples (Zero-Allocation Implementation)

### Single-Row Marginal Effect with Standard Error

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Setup model and data
df = DataFrame(y = randn(1000), x = randn(1000), z = randn(1000))
model = lm(@formula(y ~ x + z), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build derivative evaluator
vars = [:x, :z]
de = build_derivative_evaluator(compiled, data; vars=vars)
β = coef(model)
Σ = vcov(model)

# Compute marginal effect gradient for variable :x at row 1
gβ = Vector{Float64}(undef, length(compiled))
me_eta_grad_beta!(gβ, de, β, 1, :x)  # Zero allocations

# Standard error via delta method  
se = delta_method_se(gβ, Σ)  # Zero allocations
println("Marginal effect SE for x at row 1: ", se)
```

### Average Marginal Effects with Standard Errors

```julia
# AME across all observations (or subset)
rows = 1:size(df, 1)  # All rows
var = :x

# Compute AME gradient (zero-allocation with :fd backend)
gβ_ame = Vector{Float64}(undef, length(compiled))
accumulate_ame_gradient!(gβ_ame, de, β, rows, var; backend=:fd)

# Standard error for AME
se_ame = delta_method_se(gβ_ame, Σ)
println("AME standard error for x: ", se_ame)
```

### GLM with μ Marginal Effects (Logit Link)

```julia
# Logistic regression setup
df_logit = DataFrame(
    y = rand([0, 1], 1000),
    x = randn(1000), 
    z = randn(1000)
)
model_logit = glm(@formula(y ~ x + z), df_logit, Binomial(), LogitLink())
data_logit = Tables.columntable(df_logit)
compiled_logit = compile_formula(model_logit, data_logit)

de_logit = build_derivative_evaluator(compiled_logit, data_logit; vars=[:x, :z])
β_logit = coef(model_logit)
Σ_logit = vcov(model_logit)

# μ marginal effect gradient (uses chain rule)
gβ_mu = Vector{Float64}(undef, length(compiled_logit))
me_mu_grad_beta!(gβ_mu, de_logit, β_logit, 1, :x; link=LogitLink())

# Standard error for μ marginal effect
se_mu = delta_method_se(gβ_mu, Σ_logit)
println("μ marginal effect SE for x: ", se_mu)
```

### Batch Processing: Multiple Variables and Rows

```julia
# Process multiple variables efficiently
results = Dict{Symbol, Dict{String, Float64}}()
test_rows = [1, 50, 100, 500]

for var in [:x, :z]
    results[var] = Dict{String, Float64}()
    
    for (i, row) in enumerate(test_rows)
        # Single-row marginal effect gradient
        gβ = Vector{Float64}(undef, length(compiled))
        me_eta_grad_beta!(gβ, de, β, row, var)
        
        # Standard error
        se = delta_method_se(gβ, Σ)
        results[var]["row_$row"] = se
    end
    
    # AME for this variable
    gβ_ame = Vector{Float64}(undef, length(compiled))
    accumulate_ame_gradient!(gβ_ame, de, β, test_rows, var; backend=:fd)
    se_ame = delta_method_se(gβ_ame, Σ)
    results[var]["AME"] = se_ame
end

println("Standard errors: ", results)
```

### Backend Comparison

```julia
# Compare :fd vs :ad backends for η case
var = :x
test_rows = 1:20

# FD backend (zero allocations)
gβ_fd = Vector{Float64}(undef, length(compiled))
@time accumulate_ame_gradient!(gβ_fd, de, β, test_rows, var; backend=:fd)
se_fd = delta_method_se(gβ_fd, Σ)

# AD backend (small allocations, higher accuracy)  
gβ_ad = Vector{Float64}(undef, length(compiled))
@time accumulate_ame_gradient!(gβ_ad, de, β, test_rows, var; backend=:ad)
se_ad = delta_method_se(gβ_ad, Σ)

println("FD backend SE: ", se_fd)
println("AD backend SE: ", se_ad)
println("Difference: ", abs(se_fd - se_ad))
```
