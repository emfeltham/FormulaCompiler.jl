# Counterfactual Analysis with FormulaCompiler

## Overview

Counterfactual analysis examines hypothetical scenarios by systematically varying specific variables while holding others constant. This approach enables researchers to answer questions such as "What would happen if policy X were implemented?" or "How would outcomes change under different treatment assignments?"

FormulaCompiler provides an efficient override system for counterfactual analysis that achieves constant memory overhead regardless of dataset size, enabling scalable analysis of large observational datasets.

## Counterfactual Estimation in Statistical Analysis

Counterfactual analysis addresses hypothetical scenarios through controlled variable manipulation:

- **Policy impact assessment**: Evaluating minimum wage effects at different policy levels
- **Treatment effect evaluation**: Estimating outcomes under universal intervention scenarios  
- **Sensitivity analysis**: Assessing model stability across parameter ranges
- **Standardization**: Controlling for confounding variables in comparative analysis

### Research Applications

**Labor Economics**: Assess employment effects of minimum wage policies across different wage levels and regional contexts.

**Health Economics**: Evaluate treatment effects by comparing observed outcomes with scenarios where all subjects receive intervention.

**Public Finance**: Analyze tax policy impacts by modeling revenue and behavioral responses under alternative tax structures.

**Environmental Economics**: Estimate effects of carbon pricing policies across different price points and implementation scenarios.

## Computational Challenges in Counterfactual Analysis

Traditional approaches to counterfactual analysis create computational bottlenecks through data duplication:

### Memory Scaling Issues
- **Data replication**: Creating modified datasets for each scenario requires O(n × scenarios) memory
- **Storage overhead**: Large datasets become prohibitively expensive for multiple scenarios
- **Memory allocation**: Repeated data copying creates garbage collection pressure

### Performance Limitations  
- **Evaluation latency**: Model evaluation scales with dataset size for each scenario
- **Cache inefficiency**: Multiple dataset copies reduce memory locality
- **Computational redundancy**: Identical computations repeated across similar scenarios

## FormulaCompiler's Override System

FormulaCompiler addresses these limitations through a constant-memory variable substitution system:

### Core Concepts

**Scenario Construction**: Create counterfactual scenarios by specifying variable overrides
```julia
# Policy scenario: minimum wage at $15/hour
policy_scenario = create_scenario("min_wage_15", data; wage = 15.0)

# Treatment scenario: universal intervention
treatment_scenario = create_scenario("universal_treatment", data; treatment = true)
```

**Memory Efficiency**: Override system uses O(1) memory regardless of dataset size
```julia
# Original data: 1M rows × 50 variables = ~400MB
# Override scenario: adds ~48 bytes regardless of data size
scenario = create_scenario("standardized", data; age = 40, education = "HS")
```

**Zero-Allocation Evaluation**: Scenarios integrate with FormulaCompiler's compilation system
```julia
compiled = compile_formula(model, scenario.data)
output = Vector{Float64}(undef, length(compiled))
compiled(output, scenario.data, row_idx)  # 0 bytes allocated
```

### Technical Implementation

**OverrideVector**: Lazy constant vectors that return the same value for all rows
- Memory usage: Fixed overhead independent of dataset size  
- Access pattern: O(1) indexing with constant value return
- Type preservation: Maintains compatibility with original column types

**Data Integration**: Scenarios create modified data structures without copying
- Original columns: Preserved by reference (no memory duplication)
- Override columns: Replaced with OverrideVector instances
- Type stability: Maintains NamedTuple structure for FormulaCompiler compatibility

### Performance Characteristics

| Dataset Size | Data Copying | Override System |
|--------------|--------------|-----------------|
| 1K rows      | 4.8MB        | 48 bytes        |
| 100K rows    | 480MB        | 48 bytes        |
| 1M rows      | 4.8GB        | 48 bytes        |

## Basic Usage Patterns

### Single Scenario Analysis
```julia
using FormulaCompiler, GLM, Tables, DataFrames

# Fit baseline model
model = lm(@formula(wage ~ experience + education + region), data)
data_nt = Tables.columntable(data)

# Create policy counterfactual
min_wage_scenario = create_scenario("minimum_wage", data_nt; 
                                   wage = 15.0)

# Compile and evaluate
compiled = compile_formula(model, min_wage_scenario.data)
output = Vector{Float64}(undef, length(compiled))

# Evaluate for specific observations
for row in 1:10
    compiled(output, min_wage_scenario.data, row)
    predicted_wage = dot(coef(model), output)
    println("Row $row predicted wage: $predicted_wage")
end
```

### Multiple Scenario Comparison
```julia
# Create scenario grid for sensitivity analysis
wage_levels = [12.0, 15.0, 18.0, 20.0]
scenarios = [create_scenario("wage_$(w)", data_nt; wage = w) 
             for w in wage_levels]

# Evaluate scenarios for first observation
results = Float64[]
for scenario in scenarios
    compiled = compile_formula(model, scenario.data)
    output = Vector{Float64}(undef, length(compiled))
    compiled(output, scenario.data, 1)
    push!(results, dot(coef(model), output))
end

# Compare results across wage levels
for (wage, result) in zip(wage_levels, results)
    println("Wage level \$$(wage): predicted outcome = $(round(result, digits=2))")
end
```

### Categorical Variable Overrides
```julia
# Override categorical variables
education_scenario = create_scenario("college_education", data_nt;
                                    education = "College")

# Mixed continuous and categorical overrides  
standardized_scenario = create_scenario("standardized_profile", data_nt;
                                       age = 35,
                                       experience = 10,
                                       education = "High School",
                                       region = "Urban")
```

## Advanced Features

### Scenario Collections
```julia
# Systematic policy grid
policy_grid = create_scenario_grid("wage_policy", data_nt, Dict(
    :wage => [12.0, 15.0, 18.0],
    :region => ["Urban", "Rural"]
))  # Creates 6 scenarios (3 × 2 combinations)

# Evaluate entire grid
grid_results = Matrix{Float64}(undef, length(policy_grid), length(compiled))
for (i, scenario) in enumerate(policy_grid)
    compiled = compile_formula(model, scenario.data)
    for row in 1:size(grid_results, 2)
        compiled(view(grid_results, i, :), scenario.data, 1)
    end
end
```

### Performance Optimization
```julia
# Pre-compile for repeated evaluation
compiled = compile_formula(model, data_nt)
output_buffer = Vector{Float64}(undef, length(compiled))

# Create reusable scenarios
scenarios = Dict(
    "low_wage" => create_scenario("low", data_nt; wage = 10.0),
    "high_wage" => create_scenario("high", data_nt; wage = 20.0)
)

# Efficient batch evaluation
function evaluate_scenarios(scenarios, compiled, buffer, rows)
    results = Dict{String, Vector{Float64}}()
    for (name, scenario) in scenarios
        scenario_results = Float64[]
        for row in rows
            compiled(buffer, scenario.data, row)
            push!(scenario_results, dot(coef(model), buffer))
        end
        results[name] = scenario_results
    end
    return results
end
```

## Integration with Statistical Workflows

### Model Comparison
```julia
# Compare models across scenarios
models = [
    lm(@formula(y ~ x1 + x2), data),
    lm(@formula(y ~ x1 * x2), data),
    lm(@formula(y ~ x1 + x2 + x1^2), data)
]

scenario = create_scenario("test_point", data_nt; x1 = 2.0, x2 = 1.5)

for (i, model) in enumerate(models)
    compiled = compile_formula(model, scenario.data)
    output = Vector{Float64}(undef, length(compiled))
    compiled(output, scenario.data, 1)
    prediction = dot(coef(model), output)
    println("Model $i prediction: $(round(prediction, digits=3))")
end
```

### Uncertainty Quantification
```julia
# Bootstrap scenarios for uncertainty estimation
n_bootstrap = 1000
bootstrap_results = Float64[]

for _ in 1:n_bootstrap
    # Sample with replacement (implementation depends on your bootstrap method)
    boot_indices = sample(1:nrow(data), nrow(data), replace=true)
    boot_data = data[boot_indices, :]
    boot_model = lm(@formula(y ~ x1 + x2), boot_data)
    
    # Evaluate at fixed scenario
    boot_data_nt = Tables.columntable(boot_data)
    scenario = create_scenario("fixed", boot_data_nt; x1 = 2.0, x2 = 1.0)
    compiled = compile_formula(boot_model, scenario.data)
    output = Vector{Float64}(undef, length(compiled))
    compiled(output, scenario.data, 1)
    
    push!(bootstrap_results, dot(coef(boot_model), output))
end

# Compute confidence intervals
ci_lower = quantile(bootstrap_results, 0.025)
ci_upper = quantile(bootstrap_results, 0.975)
println("95% CI: [$(round(ci_lower, digits=3)), $(round(ci_upper, digits=3))]")
```

## Best Practices

### Scenario Design
- **Clear naming**: Use descriptive scenario names that indicate the counterfactual being tested
- **Systematic variation**: Create scenarios that vary one or few variables to isolate effects  
- **Realistic values**: Use empirically plausible values for override variables
- **Documentation**: Record the research question each scenario addresses

### Performance Considerations
- **Buffer reuse**: Pre-allocate output buffers and reuse across evaluations
- **Compilation caching**: Compile formulas once and reuse for multiple scenario evaluations
- **Memory monitoring**: Monitor memory usage when creating large scenario grids
- **Batch processing**: Group scenario evaluations to improve cache locality

### Statistical Interpretation  
- **Causal assumptions**: Remember that counterfactual estimates depend on modeling assumptions
- **Uncertainty**: Incorporate parameter uncertainty through bootstrap or Bayesian methods
- **Sensitivity analysis**: Test robustness across different model specifications
- **External validity**: Consider generalizability of counterfactual estimates

## Conclusion

FormulaCompiler's override system provides an efficient foundation for counterfactual analysis in statistical computing. The constant-memory approach enables scalable analysis of large datasets while maintaining compatibility with standard statistical modeling workflows.

The system's zero-allocation evaluation combined with flexible scenario specification makes it suitable for both exploratory analysis and production statistical applications requiring systematic counterfactual assessment.