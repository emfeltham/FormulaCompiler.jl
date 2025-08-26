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

---

## Function Details

### `compile_formula(model, data) -> SpecializedFormula`

Compile a statistical model formula for zero-allocation evaluation.

**Arguments:**
- `model`: A fitted statistical model (GLM, MixedModel, etc.)
- `data`: Data in Tables.jl compatible format (preferably column table)

**Returns:**
- `SpecializedFormula`: Compiled formula object for zero-allocation evaluation

**Example:**
```julia
model = lm(@formula(y ~ x + group), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)
```

### `compile_formula(compiled_complete) -> SpecializedFormula`

Convert a CompiledFormula to SpecializedFormula for maximum performance.

**Arguments:**
- `compiled_complete`: A CompiledFormula from `compile_formula_complete`

**Returns:**
- `SpecializedFormula`: High-performance compiled formula

**Example:**
```julia
complete = compile_formula_complete(model, data)
specialized = compile_formula(complete)
```

### `compile_formula_complete(model, data) -> CompiledFormula`

Create complete evaluator tree representation of the formula.

**Arguments:**
- `model`: A fitted statistical model
- `data`: Data in Tables.jl compatible format

**Returns:**
- `CompiledFormula`: Complete evaluator tree representation

**Example:**
```julia
complete = compile_formula_complete(model, data)
# Use for analysis or debugging, then specialize:
specialized = compile_formula(complete)
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

### `create_scenario_grid(name, data, parameter_dict)`

Create all combinations of scenario parameters.

**Arguments:**
- `name`: Base name for scenarios
- `data`: Base data
- `parameter_dict`: Dict mapping variables to vectors of values

**Returns:**
- `Vector{DataScenario}`: Vector of all parameter combinations

**Example:**
```julia
grid = create_scenario_grid("policy", data, Dict(
    :treatment => [false, true],
    :dose => [50, 100, 150]
))  # Creates 6 scenarios
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

#### `get_column_names(compiled_formula)`  
Get the column names that would appear in the model matrix.

#### `get_formula_terms(compiled_formula)`
Get the original formula terms used in compilation.

**Example:**
```julia
compiled = compile_formula(model, data)
n_terms = length(compiled)           # e.g., 4
col_names = get_column_names(compiled)  # ["(Intercept)", "x", "groupB", "x:groupB"]
terms = get_formula_terms(compiled)     # Original StatsModels terms
```

## Type System

### Core Types

- `SpecializedFormula`: High-performance compiled formula (zero-allocation)
- `CompiledFormula`: Complete evaluator tree representation  
- `DataScenario`: Scenario with variable overrides
- `ScenarioCollection`: Collection of related scenarios
- `OverrideVector{T}`: Memory-efficient constant vector
- `ModelRowEvaluator`: Reusable evaluator object

### Internal Types

These types are used internally but may be useful for advanced users:

- `ConstantEvaluator`: Evaluates constant terms
- `ContinuousEvaluator`: Evaluates continuous variables
- `CategoricalEvaluator`: Evaluates categorical variables  
- `FunctionEvaluator`: Evaluates function terms
- `InteractionEvaluator`: Evaluates interaction terms
- `CombinedEvaluator`: Combines multiple evaluators

## Performance Notes

- Functions marked as "zero-allocation" should show 0 bytes allocated in benchmarks
- `compile_formula` has one-time compilation cost but enables many fast evaluations
- Use `Tables.columntable` format for best performance
- Pre-allocate output vectors and reuse them across evaluations
- Batch operations with `modelrow!` are more efficient than many single evaluations