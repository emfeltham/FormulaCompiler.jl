# Advanced Features

FormulaCompiler.jl provides sophisticated features for advanced statistical computing scenarios.

## Memory-Efficient Override System

The override system allows you to create "what-if" scenarios without duplicating data in memory.

### OverrideVector

The foundation of the scenario system is `OverrideVector`, which provides a memory-efficient constant vector:

```julia
using FormulaCompiler

# Traditional approach: allocates 8MB for 1M rows
traditional = fill(42.0, 1_000_000)

# FormulaCompiler: allocates ~32 bytes
efficient = OverrideVector(42.0, 1_000_000)

# Both provide identical interface
traditional[500_000] == efficient[500_000]  # true
length(efficient) == 1_000_000              # true

# But massive memory savings
sizeof(traditional) ÷ sizeof(efficient)  # ~250,000x smaller!
```

### Data Scenarios

Create scenarios with variable overrides:

```julia
using DataFrames, Tables

df = DataFrame(
    x = randn(1000),
    y = randn(1000), 
    treatment = rand(Bool, 1000),
    group = rand(["A", "B", "C"], 1000)
)

data = Tables.columntable(df)

# Create baseline scenario
baseline = create_scenario("baseline", data)

# Create treatment scenarios
treatment_on = create_scenario("treatment_on", data; 
    treatment = true,
    dose = 100.0  # Add new variable
)

treatment_off = create_scenario("treatment_off", data;
    treatment = false,
    dose = 0.0
)

# Policy scenarios
policy_scenario = create_scenario("policy", data;
    x = mean(df.x),           # Set to population mean
    group = "A",              # Override categorical
    regulatory_flag = true    # Add policy variable
)
```

### Scenario Evaluation

Use scenarios with compiled formulas:

```julia
model = lm(@formula(y ~ x * treatment + group), df)
compiled = compile_formula(model, data)  # Compile with original data
row_vec = Vector{Float64}(undef, length(compiled))

# Evaluate different scenarios for the same individual
compiled(row_vec, baseline.data, 1)      # Original data
compiled(row_vec, treatment_on.data, 1)  # With treatment
compiled(row_vec, treatment_off.data, 1) # Without treatment
compiled(row_vec, policy_scenario.data, 1) # Policy scenario
```

### Scenario Grids

Generate all combinations of scenario parameters:

```julia
# Create comprehensive policy analysis
policy_grid = create_scenario_grid("policy_analysis", data, Dict(
    :treatment => [false, true],
    :dose => [50.0, 100.0, 150.0],
    :region => ["North", "South", "East", "West"]
))

# This creates 2×3×4 = 24 scenarios
length(policy_grid)  # 24

# Evaluate all scenarios for a specific individual
results = Matrix{Float64}(undef, length(policy_grid), length(compiled))
for (i, scenario) in enumerate(policy_grid)
    compiled(view(results, i, :), scenario.data, 1)
end

# Each row represents one scenario combination
```

### Dynamic Scenario Modification

Modify scenarios after creation:

```julia
scenario = create_scenario("dynamic", data; x = 1.0)

# Add new overrides
set_override!(scenario, :y, 100.0)
set_override!(scenario, :new_var, 42.0)

# Bulk updates
update_scenario!(scenario; 
    x = 2.0, 
    z = 0.5,
    treatment = true
)

# Remove overrides
remove_override!(scenario, :y)

# Check current overrides
get_overrides(scenario)  # Dict of current overrides
```

## Advanced Compilation Features

### Introspection and Performance

Profile compilation performance:

```julia
using BenchmarkTools

# Benchmark compilation time
@benchmark compile_formula($model, $data)
```

## Derivatives and Contrasts

Compute per-row derivatives of the model row with respect to selected variables.

ForwardDiff-based (zero-alloc after warmup):

```julia
using ForwardDiff

compiled = compile_formula(model, data)
vars = [:x, :z]  # choose continuous vars
de = build_derivative_evaluator(compiled, data; vars=vars)

J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de, 1)

# Marginal effects η = Xβ
β = coef(model)
g_eta = marginal_effects_eta(de, β, 1)  # g = J' * β

# GLM mean μ = g⁻¹(η):
using GLM
g_mu = marginal_effects_mu(de, β, 1; link=LogitLink())
```

Finite-difference fallback (simple and robust):

```julia
J_fd = derivative_modelrow_fd(compiled, data, 1; vars=vars)
```

Discrete contrasts for categorical variables:

```julia
Δ = contrast_modelrow(compiled, data, 1; var=:group3, from="A", to="B")
```

## Complex Formula Support

### Nested Functions

FormulaCompiler.jl handles complex nested functions:

```julia
@formula(y ~ log(sqrt(abs(x))) + exp(sin(z)) * group)
```

### Boolean Logic

Sophisticated boolean expressions:

```julia
@formula(y ~ (x > 0) * (z < mean(z)) + (group == "A") * log(w))
```

### Custom Functions

Define custom functions for use in formulas:

```julia
# Define custom function
custom_transform(x) = x > 0 ? log(1 + x) : -log(1 - x)

# Use in formula (requires function to be defined in scope)
@formula(y ~ custom_transform(x) + group)
```

## High-Performance Patterns

### Avoiding Allocation in Loops

```julia
# Pre-compile and pre-allocate
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))
results = Matrix{Float64}(undef, n_simulations, length(compiled))

# Monte Carlo simulation with zero allocations
for sim in 1:n_simulations
    for row in 1:nrow(df)
        compiled(row_vec, data, row)
        
        # Apply some transformation to row_vec
        results[sim, :] .= some_transformation(row_vec)
        
        # Continue processing...
    end
end
```

### Batch Processing Large Datasets

```julia
function process_large_dataset(model, data, batch_size=1000)
    compiled = compile_formula(model, data)
    n_rows = Tables.rowcount(data)
    n_cols = length(compiled)
    
    results = Vector{Matrix{Float64}}()
    
    for start_idx in 1:batch_size:n_rows
        end_idx = min(start_idx + batch_size - 1, n_rows)
        batch_size_actual = end_idx - start_idx + 1
        
        batch_result = Matrix{Float64}(undef, batch_size_actual, n_cols)
        modelrow!(batch_result, compiled, data, start_idx:end_idx)
        
        push!(results, batch_result)
    end
    
    return results
end
```

### Parallel Processing

Combine with Julia's parallel processing capabilities:

```julia
using Distributed

@everywhere using FormulaCompiler, DataFrames, Tables

function parallel_evaluation(model, data, row_indices)
    compiled = compile_formula(model, data)
    
    results = @distributed (vcat) for row_idx in row_indices
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, row_idx)
        row_vec'  # Return as row matrix
    end
    
    return results
end
```

## Integration with Optimization

### Gradient-Based Optimization

FormulaCompiler.jl works well with automatic differentiation:

```julia
using ForwardDiff

function objective_function(params, compiled_formula, data, target)
    # Update data with new parameters
    modified_data = (; data..., x = params[1], z = params[2])
    
    row_vec = Vector{Float64}(undef, length(compiled_formula))
    compiled_formula(row_vec, modified_data, 1)
    
    # Compute loss
    return sum((row_vec .- target).^2)
end

# Use with ForwardDiff for gradients
compiled = compile_formula(model, data)
target = [1.0, 2.0, 3.0, 4.0]  # Target model matrix row

gradient = ForwardDiff.gradient(
    params -> objective_function(params, compiled, data, target),
    [0.0, 1.0]  # Initial parameters
)
```

### Bayesian Analysis Integration

Efficient model evaluation in MCMC samplers:

```julia
using MCMCChains

function log_likelihood(params, compiled_formula, data, y_observed)
    # Extract parameters
    β = params[1:length(compiled_formula)]
    σ = exp(params[end])  # Log-scale for positivity
    
    n_obs = length(y_observed)
    row_vec = Vector{Float64}(undef, length(compiled_formula))
    
    ll = 0.0
    for i in 1:n_obs
        compiled_formula(row_vec, data, i)
        μ = dot(β, row_vec)
        ll += logpdf(Normal(μ, σ), y_observed[i])
    end
    
    return ll
end

# Use in MCMC sampler (pseudocode)
# sampler = MCMCSampler(log_likelihood, compiled, data, y)
```

## Memory and Performance Monitoring

### Allocation Tracking

Monitor allocation performance:

```julia
using BenchmarkTools

function check_allocations(compiled, data, n_tests=1000)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Warm up
    compiled(row_vec, data, 1)
    
    # Benchmark
    result = @benchmark begin
        for i in 1:$n_tests
            $compiled($row_vec, $data, i % nrow($data) + 1)
        end
    end
    
    return result
end

# Should show 0 allocations
benchmark_result = check_allocations(compiled, data)
```

### Memory Usage Analysis

```julia
using Profile

function profile_memory_usage(model, data, n_evaluations=10000)
    compiled = compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Profile memory
    Profile.clear_malloc_data()
    
    for i in 1:n_evaluations
        compiled(row_vec, data, i % nrow(data) + 1)
    end
    
    # Analyze results
    # (Use ProfileView.jl or similar for visualization)
end
```
