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
compiled(row_vec, data, 1)    # Zero allocations
compiled(row_vec, data, 100)  # Zero allocations
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
for i in 1:n_rows
    compiled(view(matrix, i, :), data, i)
end

# Or specific rows
specific_rows = [1, 5, 10, 50, 100]
matrix_subset = Matrix{Float64}(undef, length(specific_rows), length(compiled))
for (idx, row) in enumerate(specific_rows)
    compiled(view(matrix_subset, idx, :), data, row)
end
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

### Using DataFrames (via Tables.columntable)

Convert DataFrames to column tables once, then reuse the result:

```julia
data = Tables.columntable(df)      # Convert once
compiled = compile_formula(model, data)
compiled(row_vec, data, 1)        # Zero allocations after warmup
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

### Boolean Variables

**Boolean variables** (`Vector{Bool}`) are treated as continuous variables, matching StatsModels behavior exactly:

```julia
# Boolean data - treated as continuous
df = DataFrame(
    outcome = randn(100),
    x = randn(100),
    treated = rand(Bool, 100)  # true/false values
)

model = lm(@formula(outcome ~ x + treated), df)
compiled = compile_formula(model, Tables.columntable(df))

# Numerical encoding: false → 0.0, true → 1.0
# This matches StatsModels exactly
```

**For counterfactual analysis**, use data modification with `merge()`:

```julia
# Get dataset size
n_rows = length(data.treated)

# Boolean scenarios - individual counterfactuals
data_treated = merge(data, (treated = fill(true, n_rows),))   # All treated
compiled(output, data_treated, 1)  # Evaluate treated scenario

data_control = merge(data, (treated = fill(false, n_rows),))  # All control
compiled(output, data_control, 1)  # Evaluate control scenario

# Numeric scenarios - population analysis (70% treated probability)
data_partial = merge(data, (treated = fill(0.7, n_rows),))
compiled(output, data_partial, 1)  # Evaluate partial treatment scenario
```

**Key Points**:
- `Vector{Bool}` columns work automatically - no conversion needed
- Produces identical results to StatsModels
- Supports both boolean (`true`/`false`) and numeric (`0.7`) overrides
- Zero-allocation performance maintained

### Categorical Variables

**Required**: FormulaCompiler only supports categorical variables created with `CategoricalArrays.jl`. Raw string variables are not supported.

```julia
using CategoricalArrays

# Required: Convert string columns to categorical
df.group = categorical(df.group)
@formula(y ~ x + group)  # Automatic contrast coding

# Not supported: Raw string variables
df.category = ["A", "B", "C"]  # String vector
@formula(y ~ x + category)     # Will cause compilation errors

# Correct approach
df.category = categorical(["A", "B", "C"])
@formula(y ~ x + category)     # Works correctly
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

    # Evaluate each row in the chunk
    for i in start_idx:end_idx
        compiled(view(results, i, :), data, i)
    end
end
```

## Validation and Debugging

### Compilation Validation

Verify that compilation produces expected results:

```julia
using FormulaCompiler, GLM, Tables

# Setup test case
df = DataFrame(
    y = randn(100),
    x = randn(100), 
    group = rand(["A", "B", "C"], 100)
)
model = lm(@formula(y ~ x * group), df)
data = Tables.columntable(df)

# Compile and validate
compiled = compile_formula(model, data)

# Check dimensions
@assert length(compiled) == size(modelmatrix(model), 2) "Column count mismatch"

# Validate against GLM's modelmatrix for first few rows
mm = modelmatrix(model)
row_vec = Vector{Float64}(undef, length(compiled))

for i in 1:min(5, nrow(df))
    compiled(row_vec, data, i)
    expected = mm[i, :]
    
    if !isapprox(row_vec, expected; rtol=1e-12)
        @warn "Mismatch in row $i" row_vec expected
    else
        println("Row $i matches GLM modelmatrix")
    end
end
```

### Performance Validation

Verify zero-allocation performance:

```julia
using BenchmarkTools

# Test zero allocations
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Benchmark evaluation
result = @benchmark $compiled($row_vec, $data, 1)

# Validate performance characteristics (absolute times vary by hardware and Julia version)
@assert result.memory == 0 "Expected zero allocations, got $(result.memory) bytes"
@assert result.allocs == 0 "Expected zero allocations, got $(result.allocs) allocations"

println("Zero-allocation validation passed")
println("Memory: $(result.memory) bytes")
println("Allocations: $(result.allocs)")
```

### Data Integrity Validation

Ensure data format compatibility:

```julia
function validate_data_compatibility(model, data)
    try
        compiled = compile_formula(model, data)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        println("Data format compatible")
        return true
    catch e
        @error "Data format incompatible" exception=e
        return false
    end
end

# Test your data
is_compatible = validate_data_compatibility(model, data)
```

## Common Workflow Patterns

### Pattern 1: Monte Carlo Simulation

```julia
function monte_carlo_analysis(model, base_data, n_simulations=10000)
    compiled = compile_formula(model, base_data)
    row_vec = Vector{Float64}(undef, length(compiled))
    results = Vector{Float64}(undef, n_simulations)
    
    for i in 1:n_simulations
        # Random row selection
        random_row = rand(1:length(first(base_data)))
        compiled(row_vec, base_data, random_row)
        
        # Compute prediction (example: linear predictor)
        prediction = dot(coef(model), row_vec)
        results[i] = prediction
    end
    
    return results
end

# Usage
mc_results = monte_carlo_analysis(model, data, 10000)
println("Mean prediction: $(mean(mc_results))")
println("Std prediction: $(std(mc_results))")
```

### Pattern 2: Cross-Validation Support

```julia
function evaluate_fold(model, train_data, test_data, test_indices)
    # Compile using training data structure
    compiled = compile_formula(model, train_data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Evaluate on test data
    predictions = Vector{Float64}(undef, length(test_indices))
    
    for (i, test_row) in enumerate(test_indices)
        compiled(row_vec, test_data, test_row)
        predictions[i] = dot(coef(model), row_vec)
    end
    
    return predictions
end

# Example usage in cross-validation
train_data = Tables.columntable(df_train)
test_data = Tables.columntable(df_test)
fold_predictions = evaluate_fold(model, train_data, test_data, 1:nrow(df_test))
```

### Pattern 3: Streaming Data Processing

```julia
function process_streaming_data(model, data_stream)
    # Compile once with example data
    example_data = first(data_stream)
    compiled = compile_formula(model, example_data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    processed_results = Float64[]
    
    for data_batch in data_stream
        n_rows = length(first(data_batch))
        for row in 1:n_rows
            compiled(row_vec, data_batch, row)
            # Process each row with zero allocations
            result = dot(coef(model), row_vec)
            push!(processed_results, result)
        end
    end
    
    return processed_results
end
```

### Pattern 4: Performance-Critical Loops

```julia
function high_frequency_evaluation(model, data, row_indices)
    # Pre-compile and pre-allocate everything
    compiled = compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    results = Vector{Float64}(undef, length(row_indices))
    
    # Inner loop with zero allocations
    @inbounds for (i, row_idx) in enumerate(row_indices)
        compiled(row_vec, data, row_idx)
        # Custom computation with pre-allocated vectors
        results[i] = sum(row_vec)  # Example: sum of predictors
    end
    
    return results
end
```

## Integration with Statistical Ecosystem

### GLM.jl Integration

```julia
using GLM, Distributions

# Linear regression
linear_model = lm(@formula(y ~ x + group), df)
compiled_linear = compile_formula(linear_model, data)

# Logistic regression
df_binary = DataFrame(
    success = rand(Bool, 1000),  # Boolean response: true/false → 1.0/0.0
    x = randn(1000),
    group = rand(["A", "B"], 1000)
)
logit_model = glm(@formula(success ~ x + group), df_binary, Binomial(), LogitLink())
compiled_logit = compile_formula(logit_model, Tables.columntable(df_binary))

# Poisson regression
df_count = DataFrame(
    count = rand(Poisson(2), 1000),
    x = randn(1000),
    exposure = rand(0.5:0.1:2.0, 1000)
)
poisson_model = glm(@formula(count ~ x + log(exposure)), df_count, Poisson(), LogLink())
compiled_poisson = compile_formula(poisson_model, Tables.columntable(df_count))
```

### MixedModels.jl Integration

```julia
using MixedModels

# Mixed effects model (extracts fixed effects only)
df_mixed = DataFrame(
    y = randn(1000),
    x = randn(1000),
    treatment = rand(Bool, 1000),  # Boolean predictor: treated/untreated
    subject = rand(1:100, 1000),
    cluster = rand(1:50, 1000)
)

mixed_model = fit(MixedModel, @formula(y ~ x + treatment + (1|subject) + (1|cluster)), df_mixed)
compiled_mixed = compile_formula(mixed_model, Tables.columntable(df_mixed))

# Note: Only fixed effects (x + treatment) are compiled
# Random effects are not included in the compiled evaluator
```

### Custom Contrasts

```julia
using StatsModels

# Define custom contrast coding
contrasts_dict = Dict(
    :group => EffectsCoding(),           # Effects coding for group
    :treatment => DummyCoding(base=false) # Dummy coding with true as reference
)

model_contrasts = lm(@formula(y ~ x + group + treatment), df, contrasts=contrasts_dict)
compiled_contrasts = compile_formula(model_contrasts, data)
```

## Debugging and Troubleshooting

### Common Validation Checks

```julia
function comprehensive_validation(model, data)
    println("=== FormulaCompiler Validation ===")
    
    # 1. Compilation check
    try
        compiled = compile_formula(model, data)
        println("Compilation successful")
        println("  Formula length: $(length(compiled))")
    catch e
        println("Compilation failed: $e")
        return false
    end
    
    # 2. Zero allocation check
    compiled = compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    alloc_result = @allocated compiled(row_vec, data, 1)
    if alloc_result == 0
        println("Zero allocations achieved")
    else
        println("WARNING: Non-zero allocations: $alloc_result bytes")
    end
    
    # 3. Correctness check (first 3 rows)
    if applicable(modelmatrix, model)
        mm = modelmatrix(model)
        max_check = min(3, size(mm, 1))
        
        for i in 1:max_check
            compiled(row_vec, data, i)
            expected = mm[i, :]
            
            if isapprox(row_vec, expected; rtol=1e-12)
                println("Row $i matches reference implementation")
            else
                println("Row $i mismatch detected")
                println("  Expected: $(expected[1:min(3, length(expected))])...")
                println("  Got:      $(row_vec[1:min(3, length(row_vec))])...")
                return false
            end
        end
    end
    
    println("All validation checks passed")
    return true
end

# Run comprehensive validation
validation_result = comprehensive_validation(model, data)
```

### Performance Diagnostics

```julia
function diagnose_performance(model, data)
    println("=== Performance Diagnostics ===")
    
    # Compilation timing
    compilation_time = @elapsed compile_formula(model, data)
    println("Compilation time: $(round(compilation_time * 1000, digits=1))ms")
    
    # Setup for evaluation timing
    compiled = compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Warmup (important for accurate timing)
    for _ in 1:100
        compiled(row_vec, data, 1)
    end
    
    # Memory allocation check
    alloc_check = @allocated compiled(row_vec, data, 1)
    println("Memory allocations: $alloc_check bytes")
    
    # Performance benchmark
    bench_result = @benchmark $compiled($row_vec, $data, 1)
    println("Evaluation performance:")
    println("  Memory: $(bench_result.memory) bytes")  
    println("  Allocations: $(bench_result.allocs)")
    
    # Cache effectiveness test
    println("\nCache effectiveness:")
    cache_time_1 = @elapsed modelrow!(row_vec, model, data, 1; cache=true)
    cache_time_2 = @elapsed modelrow!(row_vec, model, data, 2; cache=true)  
    println("  First call (with compilation): $(round(cache_time_1 * 1000, digits=2))ms")
    println("  Second call (cached): $(round(cache_time_2 * 1000000, digits=1))μs")
    
    return bench_result
end

# Run diagnostics
performance_result = diagnose_performance(model, data)
```

For advanced performance optimization techniques, see the [Performance Guide](performance.md).
