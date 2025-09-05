# API Reference

Complete API reference for FormulaCompiler.jl functions and types.

## Core Compilation Functions

```@docs
compile_formula
```

## Model Row Evaluation

```@docs
modelrow
modelrow!
ModelRowEvaluator
```

## Override and Scenario System

```@docs
OverrideVector
DataScenario
create_scenario
```

## Derivatives

Dual-backend derivatives with preallocated buffers for efficiency.

### Performance Characteristics
- **Core evaluation**: 0 allocations after warmup  
- **Finite differences (FD)**: 0 allocations
- **ForwardDiff derivatives**: typically ≤512 bytes per call (ForwardDiff internals)
- **Marginal effects**: same allocation characteristics as the chosen backend
- **Validation**: Cross-validated against finite differences (rtol=1e-6, atol=1e-8)

See the Benchmark Protocol for environment and reproduction notes.

```@docs
build_derivative_evaluator
derivative_modelrow!
derivative_modelrow
derivative_modelrow_fd!
derivative_modelrow_fd
derivative_modelrow_fd_pos!
fd_jacobian_column!
fd_jacobian_column_pos!
contrast_modelrow!
contrast_modelrow
continuous_variables
marginal_effects_eta!
marginal_effects_eta
marginal_effects_eta_grad!
marginal_effects_mu!
marginal_effects_mu
me_eta_grad_beta!
me_mu_grad_beta!
delta_method_se
accumulate_ame_gradient!
```

---

## Function Details

### `compile_formula(model, data) -> UnifiedCompiled`

Compile a fitted model’s formula into a position-mapped, zero-allocation evaluator.

**Arguments:**
- `model`: Fitted statistical model (GLM, MixedModel, etc.)
- `data`: Tables.jl-compatible data (prefer a column table via `Tables.columntable`)

**Returns:**
- `UnifiedCompiled`: Type-specialized evaluator with embedded position mappings

**Example:**
```julia
model = lm(@formula(y ~ x + group), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)
```

### `modelrow(model, data, row_index) -> Vector{Float64}`

Evaluate model matrix row (allocating version).

**Arguments:**
- `model`: Fitted statistical model or compiled formula
- `data`: Data in Tables.jl format
- `row_index`: Row index to evaluate (Int) or indices (Vector{Int}/AbstractVector)

**Returns:**
- `Vector{Float64}` or `Matrix{Float64}`: Model matrix row(s)

**Example:**
```julia
row_vec = modelrow(model, data, 1)
multiple_rows = modelrow(model, data, [1, 5, 10])
```

### `modelrow!(output, compiled, data, row_indices)`

In-place model matrix row evaluation (zero-allocation).

**Arguments:**
- `output`: Pre-allocated output array (Vector or Matrix)
- `compiled`: Compiled formula object
- `data`: Data in Tables.jl format  
- `row_indices`: Row index (Int) or indices (AbstractVector)

**Example:**
```julia
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))
modelrow!(row_vec, compiled, data, 1)  # Zero allocations

# Multiple rows
matrix = Matrix{Float64}(undef, 10, length(compiled))
modelrow!(matrix, compiled, data, 1:10)
```

### `ModelRowEvaluator(model, data)`

Create a reusable model row evaluator object.

**Arguments:**
- `model`: Fitted statistical model
- `data`: Data in DataFrame or Tables.jl format

**Methods:**
- `evaluator(row_index)`: Returns new vector (allocating)
- `evaluator(output, row_index)`: In-place evaluation (non-allocating)

**Example:**
```julia
evaluator = ModelRowEvaluator(model, df)
result = evaluator(1)  # Allocating
evaluator(row_vec, 1)  # Non-allocating
```

### `create_scenario(name, data; overrides...)`

Create a data scenario with variable overrides.

**Arguments:**
- `name`: Scenario name (String)
- `data`: Base data in Tables.jl format
- `overrides...`: Keyword arguments specifying variable overrides

**Returns:**
- `DataScenario`: Scenario object with override data

**Example:**
```julia
scenario = create_scenario("treatment", data; 
    treatment = true,
    dose = 100.0
)
```

### `create_scenario_grid(name, data, parameter_dict; verbose=false)`

Create all combinations of scenario parameters.

**Arguments:**
- `name`: Base name for scenarios
- `data`: Base data
- `parameter_dict`: Dict mapping variables to vectors of values
- `verbose`: Whether to print creation progress (default: `false`)

**Returns:**
- `Vector{DataScenario}`: Vector of all parameter combinations

**Example:**
```julia
grid = create_scenario_grid("policy", data, Dict(
    :treatment => [false, true],
    :dose => [50, 100, 150]
); verbose=true)  # Creates 6 scenarios, prints progress
```

### `OverrideVector(value, length)`

Create a memory-efficient constant vector.

**Arguments:**
- `value`: Constant value to return
- `length`: Vector length

**Returns:**
- `OverrideVector`: Memory-efficient constant vector

**Example:**
```julia
# Traditional: 8MB for 1M elements
traditional = fill(42.0, 1_000_000)

# OverrideVector: ~32 bytes
efficient = OverrideVector(42.0, 1_000_000)

# Same interface
@assert traditional[500_000] == efficient[500_000]
```

### Scenario Management Functions

#### `set_override!(scenario, variable, value)`
Add or update a variable override in a scenario.

#### `remove_override!(scenario, variable)`  
Remove a variable override from a scenario.

#### `update_scenario!(scenario; overrides...)`
Bulk update multiple overrides in a scenario.

#### `get_overrides(scenario)`
Get dictionary of current overrides in a scenario.

**Example:**
```julia
scenario = create_scenario("dynamic", data)
set_override!(scenario, :x, 1.0)
update_scenario!(scenario; y = 2.0, z = 3.0)
overrides = get_overrides(scenario)  # Dict(:x => 1.0, :y => 2.0, :z => 3.0)
remove_override!(scenario, :z)
```

### Integration Functions

#### `fixed_effects_form(mixed_model)`
Extract fixed effects formula from a MixedModel.

**Arguments:**
- `mixed_model`: Fitted MixedModel

**Returns:**
- `FormulaTerm`: Fixed effects portion of the formula

**Example:**
```julia
mixed = fit(MixedModel, @formula(y ~ x + (1|group)), df)
fixed_form = fixed_effects_form(mixed)  # Returns: y ~ x
```

### Utility Functions

#### `length(compiled_formula)`
Get the number of terms in compiled formula (model matrix columns).

**Example:**
```julia
compiled = compile_formula(model, data)
n_terms = length(compiled)           # e.g., 4
```

## Categorical Mixtures

Utilities for constructing and validating categorical mixtures used in efficient profile-based marginal effects.

```@docs
mix
CategoricalMixture
MixtureWithLevels
validate_mixture_against_data
create_balanced_mixture
mixture_to_scenario_value
```

## Type System

### Core Types

- `UnifiedCompiled`: Position-mapped, zero-allocation compiled evaluator
- `DataScenario`: Scenario with variable overrides
- `ScenarioCollection`: Collection of related scenarios
- `OverrideVector{T}`: Memory-efficient constant vector
- `ModelRowEvaluator`: Reusable evaluator object

### Internal Types

Operation types used by the unified compiler:

- `LoadOp{Column, OutPos}`: Load a data column into a scratch position
- `ConstantOp{Value, OutPos}`: Place a compile-time constant into scratch
- `UnaryOp{Func, InPos, OutPos}`: Apply a unary function
- `BinaryOp{Func, InPos1, InPos2, OutPos}`: Apply a binary operation
- `ContrastOp{Column, OutPositions}`: Expand a categorical column via contrasts
- `CopyOp{InPos, OutIdx}`: Copy from scratch to final output index

### `build_derivative_evaluator(compiled, data; vars, chunk=:auto)`

Build a reusable ForwardDiff-based derivative evaluator for computing Jacobians and marginal effects.

**Arguments:**
- `compiled`: Compiled formula from `compile_formula`
- `data`: Tables.jl-compatible data (column table preferred)
- `vars`: Vector of symbols for variables to differentiate with respect to
- `chunk`: ForwardDiff chunk size (`:auto` uses `length(vars)`)

**Returns:**
- `DerivativeEvaluator`: Reusable evaluator with preallocated buffers

**Performance:**
- One-time construction cost, then ≤512 bytes per derivative call (AD backend)
- Contains preallocated Jacobian matrices and gradient vectors

**Example:**
```julia
compiled = compile_formula(model, data)
vars = continuous_variables(compiled, data)  # or [:x, :z]
de = build_derivative_evaluator(compiled, data; vars=vars)
```

<!-- Removed build_ad_evaluator: the standard build_derivative_evaluator is the supported AD entry; for zero-alloc guarantees use FD helpers. -->

### `derivative_modelrow!(J, evaluator, row)`

Fill Jacobian matrix with derivatives of model row with respect to selected variables.

**Arguments:**
- `J`: Pre-allocated matrix of size `(length(compiled), length(vars))`
- `evaluator`: `DerivativeEvaluator` from `build_derivative_evaluator`
- `row`: Row index to evaluate

**Performance:**
- ≤512 bytes allocated per call (ForwardDiff internals)
- Uses preallocated buffers for near-optimal efficiency

**Example:**
```julia
J = Matrix{Float64}(undef, length(compiled), length(de.vars))
derivative_modelrow!(J, de, 1)  # Fill J with derivatives
```

<!-- Removed zero-allocation AD variant: standard AD path is recommended and small/bounded allocations are acceptable in most cases. -->

### `marginal_effects_eta!(g, evaluator, beta, row)`

Compute marginal effects on linear predictor η = Xβ using chain rule.

**Arguments:**
- `g`: Pre-allocated gradient vector of length `length(vars)`
- `evaluator`: `DerivativeEvaluator` 
- `beta`: Model coefficients vector
- `row`: Row index

**Implementation:**
- Computes `g = J' * β` where `J` is the Jacobian matrix
- Uses preallocated internal Jacobian buffer

**Performance:**
- ≤512 bytes per call with preallocated buffers (AD backend)

**Example:**
```julia
β = coef(model)
g = Vector{Float64}(undef, length(de.vars))
marginal_effects_eta!(g, de, β, 1)
```

<!-- Removed standalone FD marginal effects: use `marginal_effects_eta!(...; backend=:fd)` -->

### `marginal_effects_eta_grad!(g, evaluator, beta, row)`

Direct gradient computation for η-scale marginal effects; pairs with parameter-gradient utilities when needed.

Example:
```julia
marginal_effects_eta_grad!(g, de, β, 1)
```

### `marginal_effects_mu!(g, evaluator, beta, row; link)`

Compute marginal effects on mean μ via chain rule: dμ/dx = (dμ/dη) × (dη/dx).

**Arguments:**
- `g`: Pre-allocated gradient vector 
- `evaluator`: `DerivativeEvaluator`
- `beta`: Model coefficients
- `row`: Row index
- `link`: GLM link function (e.g., `LogitLink()`, `LogLink()`)

**Supported Links:**
- `IdentityLink()`, `LogLink()`, `LogitLink()`, `ProbitLink()`
- `CloglogLink()`, `CauchitLink()`, `InverseLink()`, `SqrtLink()`
- `InverseSquareLink()` (when available)

**Performance:**
- ≤512 bytes per call with preallocated internal buffers (AD backend)

**Example:**
```julia
using GLM
marginal_effects_mu!(g, de, β, 1; link=LogitLink())
```

### `fd_jacobian_column!(col, evaluator, row, var)`

Compute a single column of the Jacobian matrix via finite differences, extracting derivatives with respect to one specific variable.

**Arguments:**
- `col`: Pre-allocated output vector of length `length(compiled)` (model matrix terms)
- `evaluator`: `DerivativeEvaluator` from `build_derivative_evaluator`
- `row`: Row index to evaluate (Int)
- `var`: Variable Symbol within `evaluator.vars`

**Performance:**
- 0 bytes allocated (after warmup)
- More efficient than computing full Jacobian when only one variable is needed
- Uses optimized single-variable finite difference step

**Use Cases:**
- Extract derivatives for one specific variable
- Verify ForwardDiff results per variable
- Micro-benchmark derivative computation cost per covariate
- Memory-constrained environments where full Jacobian storage is expensive

**Example:**
```julia
de = build_derivative_evaluator(compiled, data; vars=[:x, :z])
col_x = Vector{Float64}(undef, length(compiled))
fd_jacobian_column!(col_x, de, row_idx, :x)  # 0 bytes

# Compare with full Jacobian approach
J_full = Matrix{Float64}(undef, length(compiled), length(de.vars))
derivative_modelrow_fd!(J_full, de, row_idx)  # More memory, computes both variables
col_x_from_full = J_full[:, 1]  # Extract same column
```

### `fd_jacobian_column_pos!(col, evaluator, row, j)`

Positional variant of `fd_jacobian_column!` that writes directly into a positional buffer or view.

**Arguments:**
- `col`: Pre-allocated output buffer (can be a view into larger matrix)
- `evaluator`: `DerivativeEvaluator`
- `row`: Row index to evaluate (Int)
- `j`: Variable index within `evaluator.vars` (Int)

**Performance:**
- 0 bytes allocated
- Optimized for cases where output buffer is part of larger data structure
- Avoids intermediate allocations when writing to matrix views

**Use Cases:**
- Writing directly into pre-allocated matrix columns
- Integration with existing numerical pipelines that use positional buffers
- Performance-critical loops where even small allocations matter

**Example:**
```julia
# Pre-allocate matrix for storing multiple variable derivatives
n_vars = length(de.vars)
derivative_matrix = Matrix{Float64}(undef, length(compiled), n_vars)

# Fill columns one at a time using positional variant
for j in 1:n_vars
    col_view = view(derivative_matrix, :, j)
    fd_jacobian_column_pos!(col_view, de, row_idx, j)  # 0 bytes
end
```

### `me_eta_grad_beta!(gβ, evaluator, row)` and `me_mu_grad_beta!(gβ, evaluator, row; link)`

Gradients of marginal effects with respect to parameters β, for delta-method uncertainty.

Example:
```julia
gβ = Vector{Float64}(undef, length(coef(model)))
me_eta_grad_beta!(gβ, de, 1)
```

### `delta_method_se(gβ, Σ)`

Compute delta-method standard error for marginal effects using the gradient vector and parameter covariance matrix.

**Arguments:**
- `gβ`: Gradient vector of the marginal effect with respect to model parameters (Vector{Float64})
- `Σ`: Parameter covariance matrix (typically `vcov(model)`)

**Returns:**
- `Float64`: Standard error computed as `sqrt(gβ' * Σ * gβ)`

**Mathematical Foundation:**
Uses the delta method to propagate parameter uncertainty to derived quantities (marginal effects).
For a function `g(β)` with gradient `∇g(β)`, the variance is approximately `∇g(β)' * Var(β) * ∇g(β)`.

**Performance:**
- 0 bytes allocated
- Validates against negative variance (throws informative error if detected)

**Example:**
```julia
# Get parameter gradient for marginal effect at row 1
gβ = Vector{Float64}(undef, length(coef(model)))
me_eta_grad_beta!(gβ, de, 1)

# Compute standard error
se = delta_method_se(gβ, vcov(model))
println("Marginal effect SE: $se")
```

### `accumulate_ame_gradient!(gβ, evaluator, β, rows; backend=:fd)`

Accumulate the gradient of average marginal effects (AME) with respect to model parameters across multiple rows.

**Arguments:**
- `gβ`: Pre-allocated gradient accumulator vector of length `length(coef(model))`
- `evaluator`: `DerivativeEvaluator` from `build_derivative_evaluator`
- `β`: Model coefficients vector
- `rows`: Row indices to accumulate over (Vector{Int} or range)
- `backend`: Computation backend (`:fd` for finite differences, `:ad` for ForwardDiff)

**Implementation:**
Computes `gβ += ∇_β[mean_i(dη_i/dx_j)]` where the gradient is taken with respect to parameters β.
This is essential for computing standard errors of average marginal effects via the delta method.

**Performance:**
- **`:fd` backend**: 0 bytes allocated per row
- **`:ad` backend**: Small allocations per row (~400 bytes), faster computation
- Accumulates efficiently without storing individual row gradients

**Mathematical Background:**
For average marginal effects `AME = (1/n) * Σᵢ ME_i`, the parameter gradient is:
`∇_β AME = (1/n) * Σᵢ ∇_β ME_i`

**Example:**
```julia
# Initialize gradient accumulator
gβ = zeros(Float64, length(coef(model)))

# Accumulate AME gradient over rows 1-100
accumulate_ame_gradient!(gβ, de, coef(model), 1:100; backend=:fd)

# Compute average (divide by number of rows)
gβ ./= 100

# Get standard error for AME
ame_se = delta_method_se(gβ, vcov(model))
```

### `continuous_variables(compiled, data)`

Extract continuous variable names from compiled operations, excluding categoricals.

**Arguments:**
- `compiled`: Compiled formula
- `data`: Data used in compilation

**Returns:**
- `Vector{Symbol}`: Sorted list of continuous variable symbols

**Example:**
```julia
vars = continuous_variables(compiled, data)  # e.g., [:x, :z, :age]
de = build_derivative_evaluator(compiled, data; vars=vars)
```

## Override Helpers

### `create_override_data(data; overrides...)` and `create_override_vector(value, length)`

Low-level helpers for constructing scenario override containers and constant vectors. Prefer high-level `create_scenario` in user code.

Example:
```julia
ov = create_override_vector(1.0, length(data.x))
data_over = create_override_data(data; x = ov)
```

## Performance Notes

- **Core functions** (`modelrow!`, `compiled(row_vec, data, row)`) achieve exactly 0 bytes allocated
- **Derivative functions** achieve ≤512 bytes per call (ForwardDiff internals)
- **Marginal effects** use preallocated buffers to minimize allocations (≤512 bytes)
- `compile_formula` has one-time compilation cost but enables many fast evaluations
- Use `Tables.columntable` format for best performance
- Pre-allocate output vectors/matrices and reuse them across evaluations
- Build derivative evaluators once and reuse across many calls

### Single-Column vs Full Jacobian: When to Use Each

**Use `fd_jacobian_column!` when:**
- You need derivatives for only one or few specific variables
- Memory is constrained (large model matrices, many variables)
- Performing variable-by-variable sensitivity analysis
- Validating ForwardDiff results for individual covariates

**Use full Jacobian (`derivative_modelrow_fd!`) when:**
- You need derivatives for most/all variables  
- Computing marginal effects that require multiple variables
- Building gradient vectors that combine multiple partial derivatives
- Memory is abundant and computational efficiency is prioritized

**Performance Comparison:**
```julia
# Single variable: column extraction is ~3x more efficient
fd_jacobian_column!(col, de, 1, row)     # 0 bytes, minimal computation

# Multiple variables: full Jacobian amortizes setup costs  
derivative_modelrow_fd!(J, de, row)      # 0 bytes, but computes all variables
```

### AD Allocation Behavior: Standard vs Zero-Allocation Variants

FormulaCompiler provides two AD approaches with different allocation characteristics:

**Standard AD Path (Recommended for Most Users):**
- **Functions**: `build_derivative_evaluator`, `derivative_modelrow!`
- **Allocations**: Small, bounded (≤512 bytes per call)
- **Reliability**: Consistent behavior across environments
- **Performance**: Fast with predictable memory usage
- **Use case**: General-purpose derivative computation

**Zero-Allocation Options:**
- **Guaranteed zero-alloc**: Use finite differences for single-column or ME computations (e.g., `fd_jacobian_column!`, `marginal_effects_eta!(...; backend=:fd)`).
- **AD path**: Standard `derivative_modelrow!` typically allocates ≤512 bytes due to ForwardDiff internals; prefer this for accuracy and speed when tiny allocations are acceptable.

**Why Small AD Allocations Are Usually Acceptable:**
- ForwardDiff's ≤512 bytes are typically negligible compared to data processing overhead
- Small allocations are bounded and predictable (not scaling with data size)  
- Standard path provides consistent behavior across Julia/ForwardDiff versions
- Zero-allocation variants add complexity without significant benefit in most applications

**Decision Guide:**
```julia
# Standard AD (recommended for most cases)
de = build_derivative_evaluator(compiled, data; vars=vars)
derivative_modelrow!(J, de, row)  # ≤512 bytes, reliable

# Guaranteed zero allocations: FD backend
marginal_effects_eta!(g, de, β, row; backend=:fd)  # 0 bytes, always
```

### Complete Variance/SE Workflow Example

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Fit model
df = DataFrame(y = randn(1000), x = randn(1000), z = randn(1000))
model = lm(@formula(y ~ x + z), df)
data = Tables.columntable(df)

# Compile and build derivative evaluator
compiled = compile_formula(model, data)
de = build_derivative_evaluator(compiled, data; vars=[:x, :z])

# 1. Single-row marginal effect with SE
gβ_single = Vector{Float64}(undef, length(coef(model)))
me_eta_grad_beta!(gβ_single, de, 1)  # 0 bytes
se_row1 = delta_method_se(gβ_single, vcov(model))  # 0 bytes

# 2. Average marginal effect with SE
gβ_ame = zeros(Float64, length(coef(model)))
n_rows = 100
accumulate_ame_gradient!(gβ_ame, de, coef(model), 1:n_rows; backend=:fd)  # 0 bytes per row
gβ_ame ./= n_rows  # Convert to average
se_ame = delta_method_se(gβ_ame, vcov(model))  # 0 bytes

println("Single-row ME SE: $se_row1")
println("Average ME SE: $se_ame")
```
