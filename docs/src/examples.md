# Examples

Real-world examples demonstrating FormulaCompiler.jl capabilities across multiple domains.

For fundamental usage patterns, see [Basic Usage](guide/basic_usage.md). For advanced computational techniques, see [Advanced Features](guide/advanced_features.md).

## Quick Reference

Essential patterns for immediate use. All examples use synthetic data for simplicity and self-containment.

### Core Compilation Pattern
```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Basic setup
df = DataFrame(x = randn(100), y = randn(100), group = rand(["A", "B"], 100))
model = lm(@formula(y ~ x * group), df)
data = Tables.columntable(df)

# Compile and evaluate
compiled = compile_formula(model, data)
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, 1)  # Zero allocations
```

### Performance Validation
```julia
using BenchmarkTools

# Verify zero allocations
result = @benchmark $compiled($output, $data, 1)
@assert result.memory == 0 "Expected zero allocations"
```

### Scenario Creation
```julia
# Policy scenario
scenario = create_scenario("policy", data; x = 2.0, group = "A")
compiled(output, scenario.data, 1)  # Evaluate with overrides

# Scenario grid
grid = create_scenario_grid("sensitivity", data, Dict(
    :x => [-1.0, 0.0, 1.0],
    :group => ["A", "B"]
))  # Creates 6 scenarios
```

### Marginal Effects
```julia
# Identify continuous variables and build evaluator
vars = continuous_variables(compiled, data)  # [:x]
de = build_derivative_evaluator(compiled, data; vars=vars)
β = coef(model)

# Compute marginal effects
g = Vector{Float64}(undef, length(vars))
marginal_effects_eta!(g, de, β, 1; backend=:fd)  # Zero allocations
marginal_effects_eta!(g, de, β, 1; backend=:ad)  # Small allocations, higher accuracy
```

### Batch Processing
```julia
# Multiple rows at once
n_rows = 50
results = Matrix{Float64}(undef, n_rows, length(compiled))
modelrow!(results, compiled, data, 1:n_rows)  # Zero allocations
```

## Domain Applications

Real-world applications using authentic datasets from RDatasets.jl. These examples demonstrate FormulaCompiler.jl's capabilities with data structures practitioners encounter in their domains.

### Economics: Labor Market Analysis

Wage determination analysis using real Belgian wage data:

```julia
using RDatasets, FormulaCompiler, GLM, Tables

# Load real wage data from Belgian labor market
wages = dataset("Ecdat", "Bwages")  # 1472 observations
# Columns: Wage (hourly wage), Educ (education years), Exper (experience), Sex

# Convert categorical variable properly
wages.Male = wages.Sex .== "male"
data = Tables.columntable(wages)

# Wage determination model with gender interaction
model = lm(@formula(log(Wage) ~ Educ * Male + Exper + I(Exper^2)), wages)
compiled = compile_formula(model, data)
```

#### Policy Analysis with Scenarios

Analyze gender pay gap under different policy scenarios:

```julia
# Create policy scenarios
policy_scenarios = [
    create_scenario("status_quo", data),
    create_scenario("equal_education", data; Educ = mean(wages.Educ)),
    create_scenario("equal_experience", data; Exper = mean(wages.Exper)),
    create_scenario("standardized_worker", data; 
                   Educ = mean(wages.Educ), Exper = mean(wages.Exper))
]

# Analyze wage predictions across scenarios
output = Vector{Float64}(undef, length(compiled))
β = coef(model)

for scenario in policy_scenarios
    # Sample analysis for first 100 workers
    wages_predicted = Float64[]
    for i in 1:100
        compiled(output, scenario.data, i)
        push!(wages_predicted, exp(dot(β, output)))  # Convert from log scale
    end
    
    mean_wage = mean(wages_predicted)
    println("$(scenario.name): Mean predicted wage = \$$(round(mean_wage, digits=2))")
end
```

#### Marginal Effects on Wages

Compute returns to education and experience:

```julia
# Build derivative evaluator for continuous variables
continuous_vars = [:Educ, :Exper]  # Education and experience are continuous
de = build_derivative_evaluator(compiled, data; vars=continuous_vars)

# Compute marginal effects for representative workers
g_eta = Vector{Float64}(undef, length(continuous_vars))

# Male worker with median characteristics
male_median_idx = findfirst(row -> row.Male && row.Educ ≈ median(wages.Educ), eachrow(wages))
if male_median_idx !== nothing
    marginal_effects_eta!(g_eta, de, β, male_median_idx; backend=:fd)
    println("Male median worker - Returns to education: $(round(g_eta[1]*100, digits=2))%")
    println("Male median worker - Returns to experience: $(round(g_eta[2]*100, digits=2))%")
end
```

### Engineering: Automotive Performance Analysis

Engine performance modeling using the classic mtcars dataset:

```julia
using RDatasets, FormulaCompiler, GLM

# Load automotive engineering data
mtcars = dataset("datasets", "mtcars")  # 32 classic cars
# Key variables: MPG (fuel efficiency), HP (horsepower), WT (weight), Cyl (cylinders)

data = Tables.columntable(mtcars)

# Fuel efficiency model with engine characteristics
model = lm(@formula(MPG ~ HP * WT + Cyl + log(HP)), mtcars)
compiled = compile_formula(model, data)
```

#### Engineering Design Scenarios

Analyze fuel efficiency under different design specifications:

```julia
# Create engineering scenarios
base_hp = mean(mtcars.HP)
base_wt = mean(mtcars.WT)

design_scenarios = create_scenario_grid("design_analysis", data, Dict(
    :HP => [base_hp * 0.8, base_hp, base_hp * 1.2],      # -20%, baseline, +20% power
    :WT => [base_wt * 0.9, base_wt, base_wt * 1.1],      # -10%, baseline, +10% weight
    :Cyl => [4, 6, 8]                                     # Engine configurations
))

# Performance analysis across scenarios
output = Vector{Float64}(undef, length(compiled))
β = coef(model)

scenario_results = Float64[]
for scenario in design_scenarios
    compiled(output, scenario.data, 1)  # Evaluate reference car design
    predicted_mpg = dot(β, output)
    push!(scenario_results, predicted_mpg)
end

best_scenario_idx = argmax(scenario_results)
best_mpg = scenario_results[best_scenario_idx]
println("Best design scenario: $(design_scenarios[best_scenario_idx].name)")
println("Predicted MPG: $(round(best_mpg, digits=2))")
```

#### Performance Sensitivity Analysis

Compute sensitivity to design parameters:

```julia
# Marginal effects on fuel efficiency
engineering_vars = [:HP, :WT]  # Continuous engineering parameters
de = build_derivative_evaluator(compiled, data; vars=engineering_vars)

g = Vector{Float64}(undef, length(engineering_vars))

# Analyze sensitivity for different vehicle classes
for (car_type, indices) in [("Sports cars", findall(x -> x > 200, mtcars.HP)),
                            ("Economy cars", findall(x -> x < 100, mtcars.HP))]
    if !isempty(indices)
        representative_idx = indices[1]
        marginal_effects_eta!(g, de, β, representative_idx; backend=:fd)
        
        println("$car_type sensitivity:")
        println("  MPG change per HP unit: $(round(g[1], digits=3))")
        println("  MPG change per 1000lb weight: $(round(g[2]*1000, digits=3))")
    end
end
```

### Biostatistics: Cancer Survival Analysis

Clinical outcome modeling using real lung cancer data:

```julia
using RDatasets

# Load NCCTG lung cancer clinical trial data
lung = dataset("survival", "lung")  # 228 patients
# Variables: time (survival days), status (censoring), age, sex, ph.ecog (performance score)

# Remove missing values for demonstration
lung_complete = dropmissing(lung, [:age, :sex, :ph_ecog])
data = Tables.columntable(lung_complete)

# Survival time model (using log transformation for demonstration)
# In practice, would use survival analysis methods
model = lm(@formula(log(time + 1) ~ age + sex + ph_ecog + age * sex), lung_complete)
compiled = compile_formula(model, data)
```

#### Clinical Scenarios

Analyze survival outcomes under different patient profiles:

```julia
# Create clinical scenarios based on patient characteristics
median_age = median(lung_complete.age)

clinical_scenarios = [
    create_scenario("young_male", data; age = 50, sex = 1),      # Young male
    create_scenario("old_male", data; age = 70, sex = 1),        # Elderly male  
    create_scenario("young_female", data; age = 50, sex = 2),    # Young female
    create_scenario("old_female", data; age = 70, sex = 2),      # Elderly female
    create_scenario("high_performance", data; ph_ecog = 0),       # Best performance score
    create_scenario("poor_performance", data; ph_ecog = 2)        # Poor performance score
]

# Clinical outcome analysis
output = Vector{Float64}(undef, length(compiled))
β = coef(model)

for scenario in clinical_scenarios
    compiled(output, scenario.data, 1)
    predicted_log_survival = dot(β, output)
    predicted_days = exp(predicted_log_survival) - 1
    
    println("$(scenario.name): Predicted survival = $(round(predicted_days, digits=0)) days")
end
```

#### Clinical Risk Factors

Quantify impact of patient characteristics on outcomes:

```julia
# Marginal effects for continuous clinical variables
clinical_vars = [:age]  # Age is the main continuous predictor
de = build_derivative_evaluator(compiled, data; vars=clinical_vars)

g = Vector{Float64}(undef, length(clinical_vars))

# Risk assessment for different patient groups
for (group, sex_val) in [("Male patients", 1), ("Female patients", 2)]
    group_indices = findall(x -> x == sex_val, lung_complete.sex)
    if !isempty(group_indices)
        representative_idx = group_indices[div(length(group_indices), 2)]  # Median patient
        marginal_effects_eta!(g, de, β, representative_idx; backend=:fd)
        
        daily_age_effect = g[1]  # Effect per year of age
        yearly_effect = exp(daily_age_effect) - 1  # Convert from log scale
        
        println("$group - Age effect: $(round(yearly_effect*100, digits=2))% per year")
    end
end
```

### Social Sciences: Educational Outcomes

University admission analysis using UC Berkeley data:

```julia
# Load UC Berkeley admission data (famous dataset for Simpson's paradox)
ucb = dataset("datasets", "UCBAdmissions")
# This is aggregate data, so we'll expand it for modeling

# Create individual-level data from aggregate counts
individual_data = DataFrame()
for row in eachrow(ucb)
    n_cases = Int(row.Freq)
    individual_cases = DataFrame(
        Admitted = fill(row.Admit == "Admitted", n_cases),
        Gender = fill(row.Gender, n_cases), 
        Department = fill(row.Dept, n_cases)
    )
    individual_data = vcat(individual_data, individual_cases)
end

# Convert to appropriate types
individual_data.Male = individual_data.Gender .== "Male"
data = Tables.columntable(individual_data)

# Admission probability model
model = glm(@formula(Admitted ~ Male * Department), individual_data, Binomial(), LogitLink())
compiled = compile_formula(model, data)
```

#### Educational Policy Analysis

Analyze admission scenarios across departments:

```julia
# Create policy scenarios
policy_scenarios = [
    create_scenario("gender_blind", data; Male = true),    # Assume all male
    create_scenario("gender_blind_female", data; Male = false), # Assume all female
    create_scenario("dept_a_focus", data; Department = "A"),     # Focus on Dept A
    create_scenario("dept_f_focus", data; Department = "F")      # Focus on Dept F
]

# Policy impact analysis
output = Vector{Float64}(undef, length(compiled))
β = coef(model)

for scenario in policy_scenarios
    admission_probs = Float64[]
    
    # Sample 100 cases for analysis
    sample_size = min(100, nrow(individual_data))
    for i in 1:sample_size
        compiled(output, scenario.data, i)
        linear_pred = dot(β, output)
        prob = 1 / (1 + exp(-linear_pred))  # Logistic transformation
        push!(admission_probs, prob)
    end
    
    mean_prob = mean(admission_probs)
    println("$(scenario.name): Mean admission probability = $(round(mean_prob*100, digits=1))%")
end
```

## Advanced Computational Patterns

High-performance applications combining multiple FormulaCompiler.jl features with authentic datasets.

### Large-Scale Monte Carlo Simulation

Bootstrap confidence intervals for economic policy effects:

```julia
using RDatasets, Statistics, Random

# Use wage data for bootstrap analysis
wages = dataset("Ecdat", "Bwages")
wages.Male = wages.Sex .== "male"
n_obs = nrow(wages)

# Policy model
base_model = lm(@formula(log(Wage) ~ Educ * Male + Exper), wages)
base_data = Tables.columntable(wages)
compiled = compile_formula(base_model, base_data)

# Bootstrap function for policy effect estimation
function bootstrap_policy_effect(n_bootstrap=1000)
    Random.seed!(123)  # Reproducible results
    
    policy_effects = Float64[]
    output = Vector{Float64}(undef, length(compiled))
    
    for b in 1:n_bootstrap
        # Bootstrap sample
        boot_indices = rand(1:n_obs, n_obs)
        
        # Create scenarios
        status_quo = base_data
        policy_scenario = create_scenario("equal_education", base_data; 
                                        Educ = mean(wages.Educ))
        
        # Compute policy effect for bootstrap sample
        status_quo_wages = Float64[]
        policy_wages = Float64[]
        
        for idx in boot_indices[1:100]  # Sample subset for speed
            # Status quo
            compiled(output, status_quo, idx)
            push!(status_quo_wages, exp(dot(coef(base_model), output)))
            
            # Policy scenario
            compiled(output, policy_scenario.data, idx)
            push!(policy_wages, exp(dot(coef(base_model), output)))
        end
        
        # Policy effect (proportional change)
        effect = mean(policy_wages) / mean(status_quo_wages) - 1
        push!(policy_effects, effect)
    end
    
    return policy_effects
end

# Run bootstrap analysis
println("Running bootstrap analysis...")
policy_effects = bootstrap_policy_effect(500)

# Results
mean_effect = mean(policy_effects)
ci_lower = quantile(policy_effects, 0.025)
ci_upper = quantile(policy_effects, 0.975)

println("Policy effect: $(round(mean_effect*100, digits=2))%")
println("95% CI: [$(round(ci_lower*100, digits=2))%, $(round(ci_upper*100, digits=2))%]")
```

### Cross-Validation with Real Data

Model validation using automotive performance data:

```julia
using Random

# Load and prepare mtcars data
mtcars = dataset("datasets", "mtcars")
n_cars = nrow(mtcars)
data = Tables.columntable(mtcars)

function cross_validate_performance(k_folds=5)
    Random.seed!(456)
    fold_indices = rand(1:k_folds, n_cars)
    
    fold_errors = Float64[]
    output = Vector{Float64}(undef, 0)  # Will resize based on model
    
    for fold in 1:k_folds
        # Split data
        train_idx = findall(x -> x != fold, fold_indices)
        test_idx = findall(x -> x == fold, fold_indices)
        
        # Fit model on training data
        train_data = mtcars[train_idx, :]
        model = lm(@formula(MPG ~ HP + WT + Cyl), train_data)
        
        # Compile for test evaluation
        compiled = compile_formula(model, data)
        if isempty(output)
            output = Vector{Float64}(undef, length(compiled))
        end
        β = coef(model)
        
        # Predict on test set
        test_errors = Float64[]
        for test_car in test_idx
            compiled(output, data, test_car)
            predicted = dot(β, output)
            actual = mtcars.MPG[test_car]
            push!(test_errors, (predicted - actual)^2)
        end
        
        push!(fold_errors, mean(test_errors))
    end
    
    return sqrt(mean(fold_errors))  # RMSE
end

# Run cross-validation
cv_rmse = cross_validate_performance(5)
println("Cross-validation RMSE: $(round(cv_rmse, digits=2)) MPG")
```

### Parallel Scenario Analysis

Distributed policy analysis across multiple cores:

```julia
using Distributed, SharedArrays

# For demonstration - would typically use addprocs() to add workers
# addprocs(2)

# @everywhere using FormulaCompiler, RDatasets, GLM, Tables

function parallel_policy_analysis()
    wages = dataset("Ecdat", "Bwages")
    wages.Male = wages.Sex .== "male"
    data = Tables.columntable(wages)
    
    model = lm(@formula(log(Wage) ~ Educ * Male + Exper), wages)
    compiled = compile_formula(model, data)
    
    # Create comprehensive policy grid
    policy_grid = create_scenario_grid("comprehensive_policy", data, Dict(
        :Educ => [10, 12, 14, 16, 18, 20],    # Education levels
        :Exper => [0, 5, 10, 15, 20, 25],     # Experience levels  
        :Male => [false, true]                 # Gender
    ))
    
    n_scenarios = length(policy_grid)
    n_workers = min(100, nrow(wages))  # Sample size for analysis
    
    # Results storage
    results = SharedArray{Float64}(n_scenarios)
    
    # Parallel computation would go here
    # @distributed for scenario_idx in 1:n_scenarios
    for scenario_idx in 1:n_scenarios
        scenario = policy_grid[scenario_idx]
        output = Vector{Float64}(undef, length(compiled))
        β = coef(model)
        
        wage_predictions = Float64[]
        for worker in 1:n_workers
            compiled(output, scenario.data, worker)
            wage_pred = exp(dot(β, output))
            push!(wage_predictions, wage_pred)
        end
        
        results[scenario_idx] = mean(wage_predictions)
    end
    
    return results, policy_grid
end

# Run parallel analysis
println("Running comprehensive policy analysis...")
results, policy_grid = parallel_policy_analysis()

# Find optimal policy
best_idx = argmax(results)
best_scenario = policy_grid[best_idx]
best_wage = results[best_idx]

println("Optimal policy scenario:")
println("  Education: $(best_scenario.overrides[:Educ]) years")
println("  Experience: $(best_scenario.overrides[:Exper]) years") 
println("  Mean predicted wage: \$$(round(best_wage, digits=2))")
```

## Performance Notes

All examples demonstrate FormulaCompiler.jl's key performance characteristics:

- **Zero-allocation core evaluation**: `compiled(output, data, row)` calls allocate zero bytes
- **Memory-efficient scenarios**: Override system uses constant memory regardless of data size
- **Backend selection**: Choose between zero-allocation finite differences (`:fd`) and small-allocation automatic differentiation (`:ad`)
- **Scalable patterns**: Performance remains constant regardless of dataset size

For detailed performance optimization techniques, see the [Performance Guide](guide/performance.md).

## Further Reading

- [Basic Usage](guide/basic_usage.md) - Fundamental patterns and validation techniques
- [Scenario Analysis](guide/scenarios.md) - Comprehensive coverage of the override system
- [Advanced Features](guide/advanced_features.md) - Derivative computation and high-performance patterns
- [API Reference](api.md) - Complete function documentation