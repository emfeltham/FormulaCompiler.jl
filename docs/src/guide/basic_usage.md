# Basic Usage

This page covers the core functionality of FormulaCompiler.jl with practical examples.

## Core Interfaces

FormulaCompiler.jl provides three main interfaces for different use cases:

### 1. Zero-Allocation Interface (Fastest)

This is the primary interface for performance-critical applications:

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Setup
df = DataFrame(x = randn(1000), y = randn(1000), group = rand(["A", "B"], 1000))
model = lm(@formula(y ~ x * group), df)
data = Tables.columntable(df)

# Compile once
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Use many times (zero allocations)
compiled(row_vec, data, 1)    # ~50ns, 0 allocations
compiled(row_vec, data, 100)  # ~50ns, 0 allocations
```

### 2. Convenient Interface (Allocating)

For quick prototyping or when allocation isn't critical:

```julia
# Single row (returns new vector)
row_1 = modelrow(model, data, 1)

# Multiple rows (returns matrix)
rows_subset = modelrow(model, data, [1, 10, 50, 100])

# Range of rows
rows_range = modelrow(model, data, 1:10)
```

### 3. Object-Based Interface

Create a reusable evaluator object:

```julia
evaluator = ModelRowEvaluator(model, df)

# Allocating version
result = evaluator(1)

# Zero-allocation version
row_vec = Vector{Float64}(undef, length(evaluator))
evaluator(row_vec, 1)
```

## Understanding the Compiled Object

Compiled formulas contain important information:

```julia
compiled = compile_formula(model, data)

# Number of terms in the model matrix (columns)
length(compiled)  # e.g., 4 for intercept + x + group_B + x:group_B

# You can call it like a function
row_vec = Vector{Float64}(undef, length(compiled))
compiled(row_vec, data, row_index)
```

## Batch Operations

### Multiple Row Evaluation

```julia
# Pre-allocate matrix for multiple rows
n_rows = 100
matrix = Matrix{Float64}(undef, n_rows, length(compiled))

# Evaluate multiple rows efficiently
modelrow!(matrix, compiled, data, 1:n_rows)

# Or specific rows
specific_rows = [1, 5, 10, 50, 100]
matrix_subset = Matrix{Float64}(undef, length(specific_rows), length(compiled))
modelrow!(matrix_subset, compiled, data, specific_rows)
```

### Working with Views

For memory efficiency, you can work with matrix views:

```julia
big_matrix = Matrix{Float64}(undef, 1000, length(compiled))

# Fill specific rows using views
for i in 1:10
    row_view = view(big_matrix, i, :)
    compiled(row_view, data, i)
end
```

## Data Format Considerations

### Column Tables (Recommended)

For best performance, convert DataFrames to column tables:

```julia
# Convert once, reuse many times
data = Tables.columntable(df)
compiled = compile_formula(model, data)
```

### Working with DataFrames Directly

You can work with DataFrames, but column tables are more efficient:

```julia
# This works but is slower
compiled = compile_formula(model, df)
compiled(row_vec, df, 1)

# This is faster
data = Tables.columntable(df)
compiled = compile_formula(model, data)
compiled(row_vec, data, 1)
```

## Supported Formula Features

FormulaCompiler.jl handles all standard StatsModels.jl formula syntax:

### Basic Terms
```julia
# Continuous variables
@formula(y ~ x + z)

# Transformations
@formula(y ~ log(x) + sqrt(z) + x^2)

# Boolean conditions
@formula(y ~ (x > 0) + (z < mean(z)))
```

### Categorical Variables
```julia
using CategoricalArrays

df.group = categorical(df.group)
@formula(y ~ x + group)  # Automatic contrast coding
```

### Interactions
```julia
# Two-way interactions
@formula(y ~ x * group)  # Expands to: x + group + x:group

# Three-way interactions
@formula(y ~ x * y * z)

# Function interactions
@formula(y ~ log(x) * group)
```

### Complex Formulas
```julia
@formula(y ~ x * group + log(z) * treatment + sqrt(abs(w)) + (x > mean(x)))
```

## Error Handling

FormulaCompiler.jl provides clear error messages:

```julia
# Invalid row index
try
    compiled(row_vec, data, 1001)  # Only 1000 rows
catch BoundsError
    println("Row index out of bounds")
end

# Mismatched output vector size
try
    wrong_size = Vector{Float64}(undef, 3)  # Should be length(compiled)
    compiled(wrong_size, data, 1)
catch DimensionMismatch
    println("Output vector has wrong size")
end
```

## Memory Management

### Pre-allocation Best Practices

```julia
# Good: Pre-allocate and reuse
row_vec = Vector{Float64}(undef, length(compiled))
for i in 1:1000
    compiled(row_vec, data, i)
    # Process row_vec...
end

# Bad: Allocate each time
for i in 1:1000
    row_vec = modelrow(model, data, i)  # Allocates!
    # Process row_vec...
end
```

### Large Dataset Considerations

```julia
# For very large datasets, process in chunks
chunk_size = 1000
n_chunks = div(nrow(df), chunk_size)

results = Matrix{Float64}(undef, nrow(df), length(compiled))

for chunk in 1:n_chunks
    start_idx = (chunk - 1) * chunk_size + 1
    end_idx = min(chunk * chunk_size, nrow(df))
    
    chunk_results = view(results, start_idx:end_idx, :)
    modelrow!(chunk_results, compiled, data, start_idx:end_idx)
end
```

## Integration with Other Packages

FormulaCompiler.jl works seamlessly with the Julia statistical ecosystem:

```julia
using GLM, MixedModels, StandardizedPredictors

# GLM models
glm_model = glm(@formula(y ~ x), df, Normal(), IdentityLink())
compiled_glm = compile_formula(glm_model, data)

# Mixed models (extracts fixed effects only)
mixed_model = fit(MixedModel, @formula(y ~ x + (1|group)), df)
compiled_mixed = compile_formula(mixed_model, data)

# Standardized predictors
contrasts = Dict(:x => ZScore())
std_model = lm(@formula(y ~ x), df, contrasts=contrasts)
compiled_std = compile_formula(std_model, data)
```
