# Margins.jl Integration Examples

Comprehensive examples showing how FormulaCompiler's override system integrates with Margins.jl's 2×2 framework for sophisticated counterfactual analysis.

## Setup and Data Preparation

```julia
using FormulaCompiler, Margins, GLM, DataFrames, Tables, CategoricalArrays

# Example dataset: Labor economics analysis
n = 10000
df = DataFrame(
    wage = exp.(4.0 .+ 0.1 * randn(n)),
    experience = rand(0:40, n),
    education = categorical(rand(["High School", "College", "Graduate"], n)),
    region = categorical(rand(["Urban", "Suburban", "Rural"], n)),
    treatment = rand([true, false], n),
    age = rand(22:65, n),
    female = rand([true, false], n)
)

# Fit baseline model
model = lm(@formula(log(wage) ~ experience + education * region + treatment + age + female), df)
data = Tables.columntable(df)
```

## Population-Level Analysis with Overrides

### Population + Effects: Policy Impact Assessment

Evaluate average marginal effects under different policy scenarios:

```julia
# Baseline AME - using Margins.jl
baseline_ame = population_margins(model, data; type = :effects, vars = [:experience, :age])

# Universal treatment scenario
treated_scenario = create_scenario("universal_treatment", data; treatment = true)
treated_ame = population_margins(model, treated_scenario.data; 
                                type = :effects, vars = [:experience, :age])

# Compare treatment effects on marginal returns
baseline_exp_effect = baseline_ame[baseline_ame.term .== "experience", :estimate][1]
treated_exp_effect = treated_ame[treated_ame.term .== "experience", :estimate][1]
baseline_age_effect = baseline_ame[baseline_ame.term .== "age", :estimate][1]
treated_age_effect = treated_ame[treated_ame.term .== "age", :estimate][1]

experience_moderation = treated_exp_effect - baseline_exp_effect
age_moderation = treated_age_effect - baseline_age_effect

println("Experience effect moderation from treatment: $(round(experience_moderation, digits=4))")
println("Age effect moderation from treatment: $(round(age_moderation, digits=4))")

# Note: Under the hood, Margins.jl uses FormulaCompiler for efficient computation:
# - Compiles formula once with caching
# - Uses derivative_modelrow_fd! for zero-allocation derivatives
# - Aggregates across all rows for population averages
```

### Population + Predictions: Average Predictions under Policy

Assess population-level outcomes under counterfactual policies:

```julia
# Baseline average predictions - using Margins.jl
baseline_predictions = population_margins(model, data; type = :predictions, target = :mu)

# Policy scenarios
education_policy = create_scenario("education_expansion", data; education = "College")
education_predictions = population_margins(model, education_policy.data; 
                                         type = :predictions, target = :mu)

regional_policy = create_scenario("urban_development", data; region = "Urban")  
regional_predictions = population_margins(model, regional_policy.data;
                                        type = :predictions, target = :mu)

# Combined policy scenario
combined_policy = create_scenario("comprehensive_policy", data; 
                                education = "College", region = "Urban", treatment = true)
combined_predictions = population_margins(model, combined_policy.data;
                                        type = :predictions, target = :mu)

# Policy impact analysis
baseline_value = baseline_predictions.estimate[1]
education_value = education_predictions.estimate[1]
regional_value = regional_predictions.estimate[1]
combined_value = combined_predictions.estimate[1]

education_impact = (education_value - baseline_value) / baseline_value
regional_impact = (regional_value - baseline_value) / baseline_value
combined_impact = (combined_value - baseline_value) / baseline_value
synergy = combined_impact - education_impact - regional_impact

println("Education policy impact: $(round(100*education_impact, digits=2))%")
println("Regional policy impact: $(round(100*regional_impact, digits=2))%") 
println("Combined policy synergy: $(round(100*synergy, digits=2))%")

# Note: Margins.jl uses FormulaCompiler's scenario system internally for efficiency:
# - O(1) memory overhead per scenario
# - Zero-allocation evaluation with compiled formulas
# - Automatic aggregation across population
```

## Profile-Level Analysis with Controlled Conditions

### Profile + Effects: Marginal Effects with Demographic Controls

Examine marginal effects while controlling for confounding demographics:

```julia
# Standardized demographic profile
standard_demographics = create_scenario("standard_profile", data;
                                       age = 35, female = false, region = "Urban")

# MEM with standardized demographics - using Margins.jl
mem_standard = profile_margins(model, standard_demographics.data;
                             at = :means, type = :effects, vars = [:experience])

# Alternative demographic profiles
young_female = create_scenario("young_female", data; age = 25, female = true, region = "Rural")
mem_young_female = profile_margins(model, young_female.data;
                                 at = :means, type = :effects, vars = [:experience])

older_male = create_scenario("older_male", data; age = 55, female = false, region = "Suburban")  
mem_older_male = profile_margins(model, older_male.data;
                               at = :means, type = :effects, vars = [:experience])

# Compare experience returns across demographic groups
standard_return = mem_standard[mem_standard.term .== "experience", :estimate][1]
young_female_return = mem_young_female[mem_young_female.term .== "experience", :estimate][1] 
older_male_return = mem_older_male[mem_older_male.term .== "experience", :estimate][1]

println("Experience returns by demographic profile:")
println("  Standard (35, Male, Urban): $(round(100*standard_return, digits=2))%")
println("  Young Female (25, Female, Rural): $(round(100*young_female_return, digits=2))%")  
println("  Older Male (55, Male, Suburban): $(round(100*older_male_return, digits=2))%")

# Note: profile_margins() computes effects at sample means within each scenario,
# using FormulaCompiler's efficient scenario evaluation under the hood
```

### Profile + Predictions: Representative Point Predictions

Generate predictions at specific representative points:

```julia
# Representative profiles for comparison
profiles = [
    ("entry_level", create_scenario("entry", data; experience = 0, education = "High School", age = 22)),
    ("mid_career", create_scenario("mid", data; experience = 15, education = "College", age = 37)),
    ("senior_level", create_scenario("senior", data; experience = 25, education = "Graduate", age = 47))
]

# Compute predictions at specific profiles using Margins.jl
profile_predictions = []
for (name, scenario) in profiles
    # Predictions at this specific profile
    pred = profile_margins(model, scenario.data; type = :predictions, target = :mu)
    push!(profile_predictions, (profile = name, prediction = pred.estimate[1]))
end

# Display career progression predictions
println("Predicted wages by career stage:")
for (profile, prediction) in profile_predictions
    println("  $(profile): \$$(round(prediction, digits=0))")
end

# Career progression analysis
entry_wage = profile_predictions[1][2]
mid_wage = profile_predictions[2][2] 
senior_wage = profile_predictions[3][2]

early_growth = (mid_wage - entry_wage) / entry_wage
late_growth = (senior_wage - mid_wage) / mid_wage

println("Early career growth rate: $(round(100*early_growth, digits=1))%")
println("Late career growth rate: $(round(100*late_growth, digits=1))%")
```

## Advanced Integration: Systematic Policy Analysis

### Multi-Dimensional Policy Grid

Comprehensive analysis across multiple policy dimensions:

```julia
# Define policy parameter grid
policy_grid = create_scenario_grid("comprehensive_policy", data, Dict(
    :education => ["High School", "College", "Graduate"],
    :region => ["Urban", "Suburban", "Rural"], 
    :treatment => [true, false]
))  # Creates 18 scenarios (3×3×2)

# Systematic evaluation across all policy combinations
policy_results = DataFrame(
    education = String[],
    region = String[], 
    treatment = Bool[],
    ame_experience = Float64[],
    ame_age = Float64[],
    predicted_wage = Float64[]
)

for scenario in policy_grid
    # Marginal effects under this policy combination - using Margins.jl
    effects = population_margins(model, scenario.data; 
                               type = :effects, vars = [:experience, :age])
    
    # Average predictions under this policy combination - using Margins.jl
    predictions = population_margins(model, scenario.data; 
                                   type = :predictions, target = :mu)
    
    # Extract results
    exp_effect = effects[effects.term .== "experience", :estimate][1]
    age_effect = effects[effects.term .== "age", :estimate][1]
    avg_wage = predictions.estimate[1]
    
    # Store in results DataFrame
    push!(policy_results, (
        education = string(scenario.overrides[:education]),
        region = string(scenario.overrides[:region]),
        treatment = scenario.overrides[:treatment],
        ame_experience = exp_effect,
        ame_age = age_effect, 
        predicted_wage = avg_wage
    ))
end

# Note: This demonstrates the power of combining FormulaCompiler's scenario system
# with Margins.jl's statistical analysis functions - efficient O(1) scenarios
# with full-featured marginal effects computation

# Analyze policy interactions
println("Policy Analysis Results:")
println("Number of policy combinations evaluated: $(nrow(policy_results))")
println("Experience effect range: $(round(100*minimum(policy_results.ame_experience), digits=2))% to $(round(100*maximum(policy_results.ame_experience), digits=2))%")
println("Predicted wage range: \$$(round(minimum(policy_results.predicted_wage))) to \$$(round(maximum(policy_results.predicted_wage)))")

# Find optimal policy combination
optimal_idx = argmax(policy_results.predicted_wage)
optimal_policy = policy_results[optimal_idx, :]

println("Optimal policy combination:")
println("  Education: $(optimal_policy.education)")
println("  Region: $(optimal_policy.region)")  
println("  Treatment: $(optimal_policy.treatment)")
println("  Predicted wage: \$$(round(optimal_policy.predicted_wage))")
```

### Counterfactual Decomposition Analysis

Decompose policy effects into individual and interaction components:

```julia
# Baseline scenario
baseline_wage = compute_average_prediction(compiled, data, coef(model))

# Individual policy components
education_only = create_scenario("education_only", data; education = "College")
education_wage = compute_average_prediction(compiled, education_only.data, coef(model))

treatment_only = create_scenario("treatment_only", data; treatment = true)  
treatment_wage = compute_average_prediction(compiled, treatment_only.data, coef(model))

region_only = create_scenario("region_only", data; region = "Urban")
region_wage = compute_average_prediction(compiled, region_only.data, coef(model))

# Pairwise combinations
edu_treatment = create_scenario("edu_treatment", data; education = "College", treatment = true)
edu_treatment_wage = compute_average_prediction(compiled, edu_treatment.data, coef(model))

edu_region = create_scenario("edu_region", data; education = "College", region = "Urban")
edu_region_wage = compute_average_prediction(compiled, edu_region.data, coef(model))

treatment_region = create_scenario("treatment_region", data; treatment = true, region = "Urban")
treatment_region_wage = compute_average_prediction(compiled, treatment_region.data, coef(model))

# Full combination
all_policies = create_scenario("all_policies", data; 
                              education = "College", treatment = true, region = "Urban")
all_wage = compute_average_prediction(compiled, all_policies.data, coef(model))

# Decomposition analysis
main_effects = Dict(
    "education" => education_wage - baseline_wage,
    "treatment" => treatment_wage - baseline_wage,
    "region" => region_wage - baseline_wage
)

two_way_interactions = Dict(
    "education × treatment" => edu_treatment_wage - baseline_wage - 
                              main_effects["education"] - main_effects["treatment"],
    "education × region" => edu_region_wage - baseline_wage -
                           main_effects["education"] - main_effects["region"], 
    "treatment × region" => treatment_region_wage - baseline_wage -
                           main_effects["treatment"] - main_effects["region"]
)

three_way_interaction = all_wage - baseline_wage - 
                       sum(values(main_effects)) - sum(values(two_way_interactions))

# Note: In Margins.jl, this becomes:
# baseline = population_margins(model, data; type = :predictions, scale = :response)
# education_effect = population_margins(model, education_only.data; type = :predictions, scale = :response)

# Display decomposition results
println("Policy Effect Decomposition:")
println("Baseline wage: \$$(round(baseline_wage))")
println("\nMain Effects:")
for (policy, effect) in main_effects
    println("  $(policy): \$$(round(effect)) ($(round(100*effect/baseline_wage, digits=1))%)")
end

println("\nTwo-Way Interactions:")
for (interaction, effect) in two_way_interactions
    println("  $(interaction): \$$(round(effect)) ($(round(100*effect/baseline_wage, digits=1))%)")
end

println("\nThree-Way Interaction:")
println("  education × treatment × region: \$$(round(three_way_interaction)) ($(round(100*three_way_interaction/baseline_wage, digits=1))%)")

total_effect = all_effect.estimate[1] - baseline_wage
println("\nTotal Policy Effect: \$$(round(total_effect)) ($(round(100*total_effect/baseline_wage, digits=1))%)")
```

## Performance and Memory Efficiency

### Comparative Analysis: Override vs Naive Approach

Demonstrate computational advantages of override system:

```julia
using BenchmarkTools

# Scenario setup
n_scenarios = 50
wage_levels = range(10.0, 25.0, length=n_scenarios)

println("Memory and Performance Comparison:")
println("Dataset size: $(nrow(df)) observations")
println("Number of scenarios: $(n_scenarios)")

# Method 1: Override system (efficient)
override_scenarios = [create_scenario("wage_$(w)", data; wage = w) for w in wage_levels]

# Benchmark override approach
override_time = @benchmark begin
    for scenario in override_scenarios
        result = compute_average_prediction(compiled, scenario.data, coef(model))
    end
end

println("\nOverride System:")
println("  Memory usage: ~$(48 * n_scenarios) bytes")
println("  Median time: $(round(median(override_time.times) / 1e6, digits=2)) ms")

# Memory efficiency demonstration
original_data_size = Base.summarysize(data)
scenario_overhead = Base.summarysize(override_scenarios[1]) - original_data_size

println("\nMemory Analysis:")
println("  Original data: $(round(original_data_size / 1024^2, digits=2)) MB") 
println("  Scenario overhead: $(scenario_overhead) bytes")
println("  Memory efficiency: $(round(100 * (1 - scenario_overhead * n_scenarios / original_data_size), digits=4))%")
```

## Integration Best Practices Summary

### Workflow Optimization

```julia
# Efficient workflow pattern for multiple scenario analysis
function analyze_policy_scenarios(model, data, policy_params::Dict)
    # Pre-compile formula once
    compiled = FormulaCompiler.compile_formula(model, data)
    
    # Create scenarios efficiently  
    scenarios = [create_scenario("policy_$(i)", data; params...) 
                for (i, params) in enumerate(policy_params)]
    
    # Batch evaluation with pre-allocated storage
    results = []
    for scenario in scenarios
        # Reuse compiled formula with scenario data
        effects = population_margins(model, scenario.data; type = :effects)
        predictions = population_margins(model, scenario.data; type = :predictions)
        
        push!(results, (
            scenario = scenario,
            effects = effects,
            predictions = predictions
        ))
    end
    
    return results
end

# Example usage
policy_variations = [
    Dict(:treatment => true, :education => "College"),
    Dict(:treatment => false, :education => "Graduate"), 
    Dict(:treatment => true, :region => "Urban")
]

analysis_results = analyze_policy_scenarios(model, data, policy_variations)
```

The FormulaCompiler override system provides a powerful foundation for sophisticated counterfactual analysis within the Margins.jl framework, enabling efficient policy evaluation, demographic standardization, and systematic sensitivity analysis while maintaining computational efficiency and memory scalability.