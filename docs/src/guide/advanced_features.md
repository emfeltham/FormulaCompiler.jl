# Advanced Features

FormulaCompiler.jl provides sophisticated capabilities for advanced statistical computing, high-performance applications, and complex analytical workflows. This guide covers memory-efficient scenario analysis, derivative computation, and integration patterns for demanding computational environments.

## Memory-Efficient Override System

The override system allows you to create "what-if" scenarios without duplicating data in memory. For comprehensive coverage of this system, see the [Scenario Analysis](scenarios.md) guide.

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

# Massive memory savings
# OverrideVector uses constant memory regardless of length
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

## Derivative Computation System

FormulaCompiler provides comprehensive automatic differentiation capabilities for computing Jacobians, marginal effects, and gradients with dual backend support optimized for different performance requirements.

### Performance Characteristics

- **Core evaluation**: Zero allocations (modelrow!, compiled functions)
- **Finite differences (FD)**: Zero allocations (optimized implementation)
- **ForwardDiff derivatives**: Small allocations per call (ForwardDiff internals)
- **Marginal effects**: Backend-dependent allocation behavior
- **Validation**: Cross-validated against finite differences for robustness

!!! note "Backend Selection"
    FormulaCompiler provides dual backends for derivative computation: ForwardDiff (accurate with small allocations) and finite differences (zero allocations). Choose based on your performance requirements.

### Derivative Evaluator Construction

Build reusable derivative evaluators for efficient computation:

```julia
using FormulaCompiler, GLM

# Setup model with mixed variable types
df = DataFrame(
    y = randn(1000),
    price = randn(1000),          # Float64 continuous
    quantity = rand(1:100, 1000), # Int64 continuous (auto-converted)
    available = rand(Bool, 1000), # Bool continuous (true→1.0, false→0.0)
    region = categorical(rand(["North", "South"], 1000))  # Categorical
)

model = lm(@formula(y ~ price * region + log(quantity + 1) + available), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Boolean variables are treated as continuous (matching StatsModels behavior)
# available: true → 1.0, false → 0.0 in model matrix
# This produces identical results to StatsModels

# Identify continuous variables automatically
continuous_vars = continuous_variables(compiled, data)  # [:price, :quantity]

# Build derivative evaluator
de = build_derivative_evaluator(compiled, data; vars=continuous_vars)
```

### Jacobian Computation

Compute partial derivatives of model matrix rows:

```julia
# Method 1: Automatic differentiation (accurate, small allocations)
J_ad = Matrix{Float64}(undef, length(compiled), length(continuous_vars))
derivative_modelrow!(J_ad, de, 1; backend=:ad)

# Method 2: Finite differences (zero allocations)  
J_fd = Matrix{Float64}(undef, length(compiled), length(continuous_vars))
derivative_modelrow_fd_pos!(J_fd, de, 1)

# Method 3: Standalone finite differences (for validation)
J_standalone = derivative_modelrow_fd(compiled, data, 1; vars=continuous_vars)

# All methods produce equivalent results
@assert isapprox(J_ad, J_fd; rtol=1e-6) "AD and FD should match"
```

### Marginal Effects Computation

Compute effects on linear predictor and response scales:

```julia
β = coef(model)

# Effects on linear predictor η = Xβ
g_eta = Vector{Float64}(undef, length(continuous_vars))
marginal_effects_eta!(g_eta, de, β, 1; backend=:ad)  # Small allocations, accurate
marginal_effects_eta!(g_eta, de, β, 1; backend=:fd)  # Zero allocations

# Effects on response scale μ (for GLM models)
if model isa GLM.GeneralizedLinearModel
    link_function = GLM.Link(model)
    g_mu = Vector{Float64}(undef, length(continuous_vars))
    marginal_effects_mu!(g_mu, de, β, 1; link=link_function, backend=:ad)
    
    println("Marginal effects on linear predictor: $g_eta")
    println("Marginal effects on response scale: $g_mu")
end
```

### Categorical Contrasts

Analyze discrete differences for categorical variables:

```julia
# Compare categorical levels for specific row
contrast_north_south = contrast_modelrow(compiled, data, 1; 
                                       var=:region, from="North", to="South")

# Batch contrasts across multiple rows
rows_to_analyze = [1, 50, 100, 500]
contrasts = Matrix{Float64}(undef, length(rows_to_analyze), length(compiled))

for (i, row) in enumerate(rows_to_analyze)
    contrast = contrast_modelrow(compiled, data, row; var=:region, from="North", to="South")
    contrasts[i, :] .= contrast
end
```

### Advanced Configuration

Optimize derivative computation for specific use cases:

```julia
# Variable selection strategies
all_continuous = continuous_variables(compiled, data)
economic_vars = [:price, :quantity]  # Domain-specific subset
interaction_vars = [:price]          # Focus on key interactions

# Chunking for large variable sets  
large_var_set = [:var1, :var2, :var3, :var4, :var5, :var6, :var7, :var8]
de_chunked = build_derivative_evaluator(compiled, data; 
                                       vars=large_var_set, 
                                       chunk=ForwardDiff.Chunk{4}())  # Process in chunks of 4

# Backend selection based on requirements
function compute_derivatives_with_backend_choice(de, β, row, require_zero_alloc=false)
    backend = require_zero_alloc ? :fd : :ad
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta!(g, de, β, row; backend=backend)
    return g
end
```

### Mixed Models (Fixed Effects)

Derivatives target the fixed-effects design (random effects are intentionally excluded):

```julia
using MixedModels

df = DataFrame(y = randn(500), x = randn(500), z = abs.(randn(500)) .+ 0.1,
               group = categorical(rand(1:20, 500)))
mm = fit(MixedModel, @formula(y ~ 1 + x + z + (1|group)), df; progress=false)

data = Tables.columntable(df)
compiled = compile_formula(mm, data)  # fixed-effects only
vars = [:x, :z]
de = build_derivative_evaluator(compiled, data; vars=vars)

J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de, 1)
```

### Architecture and Optimization

The derivative system achieves near-zero allocations through:

- **Preallocated buffers**: Jacobian matrices, gradient vectors, and temporary arrays stored in `DerivativeEvaluator`
- **Typed closures**: Compile-time specialization eliminates runtime dispatch
- **Prebuilt data structures**: Override vectors and merged data reused across calls
- **Optimized memory layout**: All allocations front-loaded during evaluator construction

### Performance Benchmarking

```julia
using BenchmarkTools

# Build evaluator once (one-time cost)
de = build_derivative_evaluator(compiled, data; vars=[:x, :z])
J = Matrix{Float64}(undef, length(compiled), length(de.vars))

# Benchmark derivatives
@benchmark derivative_modelrow!($J, $de, 25)

# Benchmark marginal effects
β = coef(model)
g = Vector{Float64}(undef, length(de.vars))
@benchmark marginal_effects_eta!($g, $de, $β, 25; backend=:ad)  # Small allocations
@benchmark marginal_effects_eta!($g, $de, $β, 25; backend=:fd)  # Zero allocations
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

### Categorical Mixtures

FormulaCompiler.jl supports categorical mixtures for marginal effects computation:

```julia
# Create data with weighted categorical specifications
df = DataFrame(
    x = [1.0, 2.0, 3.0],
    group = [mix("A" => 0.3, "B" => 0.7),   # 30% A, 70% B
             mix("A" => 0.3, "B" => 0.7),
             mix("A" => 0.3, "B" => 0.7)]
)

# Compile and evaluate with zero allocations
compiled = compile_formula(model, Tables.columntable(df))
compiled(output, Tables.columntable(df), 1)  # ~50ns, 0 bytes
```

For comprehensive coverage of categorical mixtures including validation, helper functions, and marginal effects integration, see the [Categorical Mixtures](categorical_mixtures.md) guide.

## High-Performance Computing Patterns

### Zero-Allocation Computational Loops

Design computational patterns that maintain zero-allocation performance:

```julia
function monte_carlo_predictions(model, data, n_simulations=10000)
    # Pre-compile and pre-allocate all necessary buffers
    compiled = compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    β = coef(model)
    
    # Pre-allocate result storage
    predictions = Vector{Float64}(undef, n_simulations)
    data_size = length(first(data))
    
    # Zero-allocation simulation loop
    for sim in 1:n_simulations
        # Random row selection
        row_idx = rand(1:data_size)
        
        # Zero-allocation model matrix evaluation
        compiled(row_vec, data, row_idx)
        
        # Zero-allocation prediction computation
        predictions[sim] = dot(β, row_vec)
    end
    
    return predictions
end

# Usage with performance validation
predictions = monte_carlo_predictions(model, data, 100000)
```

### Advanced Memory Management

Optimize memory usage for large-scale applications:

```julia
function memory_efficient_batch_processing(model, large_dataset, batch_size=1000)
    compiled = compile_formula(model, large_dataset)
    n_total = length(first(large_dataset))
    n_batches = cld(n_total, batch_size)
    
    # Pre-allocate reusable buffers
    row_vec = Vector{Float64}(undef, length(compiled))
    batch_results = Matrix{Float64}(undef, batch_size, length(compiled))
    
    all_results = Vector{Matrix{Float64}}()
    
    for batch in 1:n_batches
        start_idx = (batch - 1) * batch_size + 1
        end_idx = min(batch * batch_size, n_total)
        actual_batch_size = end_idx - start_idx + 1
        
        # Resize for last batch if needed
        if actual_batch_size != batch_size
            batch_results = Matrix{Float64}(undef, actual_batch_size, length(compiled))
        end
        
        # Zero-allocation batch evaluation
        for (local_idx, global_idx) in enumerate(start_idx:end_idx)
            compiled(view(batch_results, local_idx, :), large_dataset, global_idx)
        end
        
        # Store results (could write to disk here for very large datasets)
        push!(all_results, copy(batch_results))
    end
    
    return all_results
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
        compiled(row_vec, data, i % length(first(data)) + 1)
    end
    
    # Analyze results
    # (Use ProfileView.jl or similar for visualization)
end
```

## Real-World Application Patterns

### Economic Policy Analysis

Combine scenario analysis with derivative computation for comprehensive policy evaluation:

```julia
function policy_impact_analysis(baseline_model, policy_data, policy_parameters)
    # Compile baseline model
    compiled = compile_formula(baseline_model, policy_data)
    β = coef(baseline_model)
    
    # Identify continuous policy levers
    policy_vars = intersect(keys(policy_parameters), continuous_variables(compiled, policy_data))
    de = build_derivative_evaluator(compiled, policy_data; vars=collect(policy_vars))
    
    # Create policy scenarios
    scenarios = [
        create_scenario("status_quo", policy_data),
        create_scenario("moderate_policy", policy_data; policy_parameters...),
        create_scenario("aggressive_policy", policy_data; 
                       [k => v * 1.5 for (k, v) in policy_parameters]...)
    ]
    
    # Evaluate policy impacts
    n_individuals = min(1000, length(first(policy_data)))  # Sample for analysis
    
    results = Dict()
    for scenario in scenarios
        scenario_predictions = Vector{Float64}(undef, n_individuals)
        scenario_marginals = Matrix{Float64}(undef, n_individuals, length(policy_vars))
        
        # Evaluate predictions and marginal effects for each individual
        row_vec = Vector{Float64}(undef, length(compiled))
        marginal_vec = Vector{Float64}(undef, length(policy_vars))
        
        for i in 1:n_individuals
            # Prediction
            compiled(row_vec, scenario.data, i)
            scenario_predictions[i] = dot(β, row_vec)
            
            # Marginal effects
            marginal_effects_eta!(marginal_vec, de, β, i; backend=:fd)  # Zero allocations
            scenario_marginals[i, :] .= marginal_vec
        end
        
        results[scenario.name] = (
            predictions = scenario_predictions,
            marginal_effects = scenario_marginals,
            mean_prediction = mean(scenario_predictions),
            policy_sensitivity = mean(scenario_marginals, dims=1)
        )
    end
    
    return results
end

# Example usage
policy_params = Dict(:minimum_wage => 15.0, :tax_rate => 0.25)
analysis_results = policy_impact_analysis(economic_model, economic_data, policy_params)

# Compare scenarios
status_quo_mean = analysis_results["status_quo"].mean_prediction
moderate_mean = analysis_results["moderate_policy"].mean_prediction
policy_effect = moderate_mean - status_quo_mean

println("Policy effect: $(round(policy_effect, digits=3))")
```

### Biostatistical Applications

High-throughput analysis for medical and biological research:

```julia
function biomarker_analysis(survival_model, patient_data, biomarker_ranges)
    compiled = compile_formula(survival_model, patient_data)
    β = coef(survival_model)
    
    # Identify biomarker variables for sensitivity analysis
    biomarker_vars = Symbol.(keys(biomarker_ranges))
    continuous_biomarkers = intersect(biomarker_vars, continuous_variables(compiled, patient_data))
    
    if !isempty(continuous_biomarkers)
        de = build_derivative_evaluator(compiled, patient_data; vars=continuous_biomarkers)
    end
    
    # Create biomarker scenarios
    biomarker_scenarios = create_scenario_grid("biomarker_analysis", patient_data, biomarker_ranges)
    
    # Patient risk stratification
    n_patients = length(first(patient_data))
    risk_matrix = Matrix{Float64}(undef, length(biomarker_scenarios), n_patients)
    
    # Pre-allocate evaluation buffers
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Evaluate all scenario-patient combinations
    for (scenario_idx, scenario) in enumerate(biomarker_scenarios)
        for patient_idx in 1:n_patients
            compiled(row_vec, scenario.data, patient_idx)
            
            # Compute risk score (example: linear predictor)
            risk_score = dot(β, row_vec)
            risk_matrix[scenario_idx, patient_idx] = risk_score
        end
    end
    
    # Compute marginal effects for sensitivity analysis
    if !isempty(continuous_biomarkers)
        marginal_matrix = Matrix{Float64}(undef, n_patients, length(continuous_biomarkers))
        marginal_vec = Vector{Float64}(undef, length(continuous_biomarkers))
        
        for patient_idx in 1:n_patients
            marginal_effects_eta!(marginal_vec, de, β, patient_idx; backend=:fd)
            marginal_matrix[patient_idx, :] .= marginal_vec
        end
        
        return (
            risk_scores = risk_matrix,
            marginal_effects = marginal_matrix,
            scenarios = biomarker_scenarios,
            biomarker_vars = continuous_biomarkers
        )
    else
        return (
            risk_scores = risk_matrix,
            scenarios = biomarker_scenarios
        )
    end
end

# Example usage
biomarker_ranges = Dict(
    :tumor_size => [1.0, 2.0, 3.0, 4.0],     # cm
    :psa_level => [4.0, 10.0, 20.0],         # ng/mL
    :age => [50, 65, 80]                     # years
)

bio_results = biomarker_analysis(oncology_model, patient_data, biomarker_ranges)
```

### Financial Risk Modeling

Scenario analysis for financial applications:

```julia
function portfolio_risk_analysis(risk_model, market_data, stress_scenarios)
    compiled = compile_formula(risk_model, market_data)
    β = coef(risk_model)
    
    # Risk factor sensitivity
    risk_factors = continuous_variables(compiled, market_data)
    if !isempty(risk_factors)
        de = build_derivative_evaluator(compiled, market_data; vars=risk_factors)
    end
    
    # Create market stress scenarios
    stress_scenario_objects = [
        create_scenario(name, market_data; parameters...)
        for (name, parameters) in stress_scenarios
    ]
    
    # Portfolio evaluation across scenarios
    n_assets = length(first(market_data))
    scenario_valuations = Dict{String, Vector{Float64}}()
    
    row_vec = Vector{Float64}(undef, length(compiled))
    
    for scenario in stress_scenario_objects
        asset_valuations = Vector{Float64}(undef, n_assets)
        
        for asset_idx in 1:n_assets
            compiled(row_vec, scenario.data, asset_idx)
            # Risk-adjusted valuation
            asset_valuations[asset_idx] = dot(β, row_vec)
        end
        
        scenario_valuations[scenario.name] = asset_valuations
    end
    
    # Risk sensitivity analysis
    if !isempty(risk_factors)
        sensitivity_matrix = Matrix{Float64}(undef, n_assets, length(risk_factors))
        sensitivity_vec = Vector{Float64}(undef, length(risk_factors))
        
        for asset_idx in 1:n_assets
            marginal_effects_eta!(sensitivity_vec, de, β, asset_idx; backend=:fd)
            sensitivity_matrix[asset_idx, :] .= sensitivity_vec
        end
        
        return (
            scenario_valuations = scenario_valuations,
            risk_sensitivities = sensitivity_matrix,
            risk_factors = risk_factors
        )
    else
        return (scenario_valuations = scenario_valuations,)
    end
end

# Example usage
stress_scenarios = [
    ("market_crash", Dict(:market_volatility => 0.4, :interest_rate => 0.02)),
    ("inflation_shock", Dict(:inflation_rate => 0.08, :commodity_index => 1.5)),
    ("recession", Dict(:gdp_growth => -0.03, :unemployment => 0.12))
]

risk_analysis = portfolio_risk_analysis(financial_model, market_data, stress_scenarios)

# Analyze results
baseline_value = sum(risk_analysis.scenario_valuations["baseline"])
crash_value = sum(risk_analysis.scenario_valuations["market_crash"])
portfolio_risk = (crash_value - baseline_value) / baseline_value

println("Portfolio stress loss: $(round(portfolio_risk * 100, digits=2))%")
```

## Further Reading

- [Scenario Analysis Guide](scenarios.md) - Comprehensive coverage of the override system
- [Performance Guide](performance.md) - Detailed optimization strategies and benchmarking
- [Examples](../examples.md) - Additional domain-specific applications
- [API Reference](../api.md) - Complete function documentation
