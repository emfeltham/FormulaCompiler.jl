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
build_zero_alloc_ad_evaluator
derivative_modelrow!
derivative_modelrow
derivative_modelrow_zero_alloc!
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
marginal_effects_eta_fd!
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

### `build_zero_alloc_ad_evaluator(compiled, data; vars)`

Build an AD-based derivative evaluator configured to minimize allocations further than the default (environment-permitting).

Notes:
- Preserves accuracy of ForwardDiff; may reduce transient allocations below typical ≤512 bytes depending on environment.
- Falls back gracefully to standard AD behavior when zero allocation cannot be guaranteed.

Example:
```julia
de_za = build_zero_alloc_ad_evaluator(compiled, data; vars=[:x, :z])
```

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

### `derivative_modelrow_zero_alloc!(J, evaluator, row)`

AD Jacobian with additional zero-allocation safeguards (where supported).

Performance:
- Matches `derivative_modelrow!` numerics; aims to reduce transient allocations beyond ForwardDiff defaults.

Example:
```julia
derivative_modelrow_zero_alloc!(J, de_za, 1)
```

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

### `marginal_effects_eta_fd!(g, evaluator, beta, row)`

Finite-difference marginal effects on η; zero-allocation path for reproducible performance verification.

Example:
```julia
marginal_effects_eta_fd!(g, de, β, 1)
```

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

### `fd_jacobian_column!(col, evaluator, j, row)` and `fd_jacobian_column_pos!(col, evaluator, j, row)`

Compute a single Jacobian column via finite differences; the `_pos!` variant writes into a positional view/buffer.

Use cases:
- Verify AD results per variable; micro-benchmark derivative cost per covariate.

Example:
```julia
fd_jacobian_column!(col_buf, de, j, 1)
```

### `me_eta_grad_beta!(gβ, evaluator, row)` and `me_mu_grad_beta!(gβ, evaluator, row; link)`

Gradients of marginal effects with respect to parameters β, for delta-method uncertainty.

Example:
```julia
gβ = Vector{Float64}(undef, length(coef(model)))
me_eta_grad_beta!(gβ, de, 1)
```

### `delta_method_se(gβ, Σ)`

Compute delta-method standard error given parameter-gradient `gβ` and covariance matrix `Σ`.

Returns:
- `Float64`: Standard error `sqrt(gβ' * Σ * gβ)`

Example:
```julia
se = delta_method_se(gβ, vcov(model))
```

### `accumulate_ame_gradient!(gβ, evaluator, β, rows; backend=:fd)`

Accumulate average marginal effect parameter-gradient across a set of rows; supports `:fd` or `:ad` backends.

Example:
```julia
accumulate_ame_gradient!(gβ, de, β, 1:100; backend=:fd)
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
