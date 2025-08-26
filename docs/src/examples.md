# Examples

Real-world examples demonstrating FormulaCompiler.jl in action.

## Monte Carlo Simulation

High-performance Monte Carlo simulation using zero-allocation evaluation:

```julia
using FormulaCompiler, GLM, DataFrames, Tables, Random
using BenchmarkTools, Statistics

function monte_carlo_simulation(n_sims=1_000_000, n_obs=1000)
    # Generate base dataset
    Random.seed!(123)
    df = DataFrame(
        y = randn(n_obs),
        x = randn(n_obs),
        z = abs.(randn(n_obs)) .+ 0.1,
        treatment = rand(Bool, n_obs),
        group = categorical(rand(["A", "B", "C"], n_obs))
    )
    
    # Fit model
    model = lm(@formula(y ~ x * treatment + log(z) + group), df)
    data = Tables.columntable(df)
    
    # Compile for zero-allocation evaluation
    compiled = compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Pre-allocate results
    results = Vector{Float64}(undef, n_sims)
    
    println("Running $n_sims Monte Carlo simulations...")
    @time begin
        for sim in 1:n_sims
            # Random row selection
            row_idx = rand(1:n_obs)
            
            # Zero-allocation evaluation
            compiled(row_vec, data, row_idx)
            
            # Calculate linear predictor
            results[sim] = dot(coef(model), row_vec)
        end
    end
    
    return results
end

# Run simulation
mc_results = monte_carlo_simulation(1_000_000, 1000)

println("Monte Carlo Results:")
println("Mean: $(round(mean(mc_results), digits=4))")
println("Std:  $(round(std(mc_results), digits=4))")
println("Min:  $(round(minimum(mc_results), digits=4))")  
println("Max:  $(round(maximum(mc_results), digits=4))")
```

## Bootstrap Confidence Intervals

Efficient bootstrap resampling for coefficient confidence intervals:

```julia
using Random

function bootstrap_confidence_intervals(model, data, n_bootstrap=1000, confidence=0.95)
    compiled = compile_formula(model, data)
    n_obs = Tables.rowcount(data)
    n_coefs = length(compiled)
    
    # Get response variable
    y_name = Symbol(model.mf.f.lhs)
    y = data[y_name]
    
    # Pre-allocate
    bootstrap_coefs = Matrix{Float64}(undef, n_bootstrap, n_coefs)
    row_vec = Vector{Float64}(undef, n_coefs)
    X_bootstrap = Matrix{Float64}(undef, n_obs, n_coefs)
    y_bootstrap = Vector{Float64}(undef, n_obs)
    
    println("Computing $n_bootstrap bootstrap samples...")
    @time begin
        for b in 1:n_bootstrap
            # Generate bootstrap sample indices
            sample_indices = rand(1:n_obs, n_obs)
            
            # Build design matrix for bootstrap sample (zero allocations)
            for (i, idx) in enumerate(sample_indices)
                compiled(row_vec, data, idx)
                X_bootstrap[i, :] .= row_vec
                y_bootstrap[i] = y[idx]
            end
            
            # Compute bootstrap coefficients
            bootstrap_coefs[b, :] = X_bootstrap \ y_bootstrap
        end
    end
    
    # Calculate confidence intervals
    α = 1 - confidence
    lower_percentile = 100 * (α/2)
    upper_percentile = 100 * (1 - α/2)
    
    coef_names = coefnames(model)
    original_coefs = coef(model)
    
    println("\n$(Int(confidence*100))% Bootstrap Confidence Intervals:")
    println("─"^60)
    
    for (i, name) in enumerate(coef_names)
        boot_coefs_i = bootstrap_coefs[:, i]
        lower = percentile(boot_coefs_i, lower_percentile)
        upper = percentile(boot_coefs_i, upper_percentile)
        
        println("$name:")
        println("  Original: $(round(original_coefs[i], digits=4))")
        println("  Bootstrap: [$(round(lower, digits=4)), $(round(upper, digits=4))]")
        println("  Bootstrap SE: $(round(std(boot_coefs_i), digits=4))")
    end
    
    return bootstrap_coefs
end

# Example usage
df = DataFrame(
    y = randn(500),
    x = randn(500),
    treatment = rand(Bool, 500),
    age = rand(20:80, 500)
)

model = lm(@formula(y ~ x * treatment + age), df)
data = Tables.columntable(df)

boot_coefs = bootstrap_confidence_intervals(model, data, 1000, 0.95)
```

## Policy Impact Analysis

Comprehensive policy scenario analysis using the scenario system:

```julia
function policy_impact_analysis()
    # Simulate policy-relevant dataset
    Random.seed!(456)
    n_individuals = 10000
    
    df = DataFrame(
        # Outcome: earnings (thousands)
        earnings = max.(0, randn(n_individuals) * 15 .+ 45),
        
        # Demographics
        age = rand(22:65, n_individuals),
        education_years = rand(10:18, n_individuals),
        experience = rand(0:30, n_individuals),
        
        # Geographic
        region = categorical(rand(["North", "South", "East", "West"], n_individuals)),
        urban = rand(Bool, n_individuals),
        
        # Current programs
        job_training = rand(Bool, n_individuals),
        healthcare_access = rand(Bool, n_individuals)
    )
    
    # Fit earnings model
    model = lm(@formula(earnings ~ age + education_years + experience + 
                                 region + urban + job_training * healthcare_access), df)
    
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    
    # Define policy scenarios
    scenarios = Dict(
        "status_quo" => create_scenario("status_quo", data),
        
        "universal_training" => create_scenario("universal_training", data;
            job_training = true
        ),
        
        "universal_healthcare" => create_scenario("universal_healthcare", data;
            healthcare_access = true  
        ),
        
        "combined_programs" => create_scenario("combined", data;
            job_training = true,
            healthcare_access = true
        ),
        
        "education_boost" => create_scenario("education_boost", data;
            education_years = mean(df.education_years) + 2  # +2 years education
        ),
        
        "comprehensive_policy" => create_scenario("comprehensive", data;
            job_training = true,
            healthcare_access = true, 
            education_years = mean(df.education_years) + 1,
            urban = true  # Urbanization investment
        )
    )
    
    # Evaluate policy impacts
    results = Dict{String, NamedTuple}()
    row_vec = Vector{Float64}(undef, length(compiled))
    
    println("Policy Impact Analysis Results:")
    println("="^50)
    
    for (policy_name, scenario) in scenarios
        predictions = Vector{Float64}(undef, n_individuals)
        
        # Calculate predictions for all individuals under this policy
        for i in 1:n_individuals
            compiled(row_vec, scenario.data, i)
            predictions[i] = dot(coef(model), row_vec)
        end
        
        # Calculate impacts vs status quo
        if policy_name != "status_quo"
            status_quo_preds = Vector{Float64}(undef, n_individuals)
            for i in 1:n_individuals
                compiled(row_vec, scenarios["status_quo"].data, i)
                status_quo_preds[i] = dot(coef(model), row_vec)
            end
            
            individual_impacts = predictions .- status_quo_preds
            
            results[policy_name] = (
                mean_earnings = mean(predictions),
                mean_impact = mean(individual_impacts),
                median_impact = median(individual_impacts),
                impact_std = std(individual_impacts),
                percent_helped = 100 * mean(individual_impacts .> 0),
                total_cost = sum(max.(0, individual_impacts)) * 1000  # Total $ impact
            )
        else
            results[policy_name] = (
                mean_earnings = mean(predictions),
                mean_impact = 0.0,
                median_impact = 0.0,
                impact_std = 0.0,
                percent_helped = 0.0,
                total_cost = 0.0
            )
        end
    end
    
    # Display results
    baseline_earnings = results["status_quo"].mean_earnings
    
    for (policy, stats) in results
        if policy == "status_quo"
            println("$policy (baseline):")
            println("  Mean earnings: \$$(round(Int, stats.mean_earnings))k")
        else
            println("\n$policy:")
            println("  Mean earnings: \$$(round(Int, stats.mean_earnings))k")
            println("  Mean impact: \$$(round(Int, stats.mean_impact))k per person")
            println("  Median impact: \$$(round(Int, stats.median_impact))k per person") 
            println("  % individuals helped: $(round(stats.percent_helped, digits=1))%")
            println("  Total economic impact: \$$(round(Int, stats.total_cost/1000))M")
            
            # Cost-effectiveness (simplified)
            if policy == "universal_training"
                cost_per_person = 5000  # $5k per person for training
            elseif policy == "universal_healthcare"
                cost_per_person = 8000  # $8k per person for healthcare
            elseif policy == "combined_programs"  
                cost_per_person = 12000  # $12k for both
            elseif policy == "education_boost"
                cost_per_person = 15000  # $15k for education
            else
                cost_per_person = 20000  # $20k comprehensive
            end
            
            total_program_cost = cost_per_person * n_individuals / 1000  # In thousands
            net_benefit = stats.total_cost - total_program_cost
            roi = (net_benefit / total_program_cost) * 100
            
            println("  Program cost: \$$(round(Int, total_program_cost/1000))M")
            println("  Net benefit: \$$(round(Int, net_benefit/1000))M")
            println("  ROI: $(round(roi, digits=1))%")
        end
    end
    
    return results, scenarios
end

# Run the analysis
policy_results, policy_scenarios = policy_impact_analysis()
```

## Marginal Effects Calculation

Efficient numerical derivatives for marginal effects:

```julia
function marginal_effects_analysis(model, data, variables=nothing; delta=0.01)
    compiled = compile_formula(model, data)
    n_obs = Tables.rowcount(data)
    n_coefs = length(compiled)
    
    # Default to all continuous variables
    if variables === nothing
        variables = [:age, :experience, :education_years]  # Adjust as needed
    end
    
    results = Dict{Symbol, Matrix{Float64}}()
    row_vec_orig = Vector{Float64}(undef, n_coefs)
    row_vec_pert = Vector{Float64}(undef, n_coefs)
    
    for var in variables
        println("Computing marginal effects for $var...")
        
        # Get original values
        original_values = data[var]
        perturbed_values = original_values .+ delta
        perturbed_data = (; data..., var => perturbed_values)
        
        marginal_effects = Matrix{Float64}(undef, n_obs, n_coefs)
        
        @time begin
            for i in 1:n_obs
                # Original prediction
                compiled(row_vec_orig, data, i)
                
                # Perturbed prediction  
                compiled(row_vec_pert, perturbed_data, i)
                
                # Marginal effects for each coefficient
                marginal_effects[i, :] .= (row_vec_pert .- row_vec_orig) ./ delta
            end
        end
        
        results[var] = marginal_effects
    end
    
    # Summarize results
    coef_names = coefnames(model)
    println("\nMarginal Effects Summary:")
    println("="^40)
    
    for var in variables
        println("\nVariable: $var")
        me_matrix = results[var]
        
        for (j, coef_name) in enumerate(coef_names)
            me_col = me_matrix[:, j]
            println("  $coef_name:")
            println("    Mean ME: $(round(mean(me_col), digits=6))")
            println("    Std ME:  $(round(std(me_col), digits=6))")
            println("    Range:   [$(round(minimum(me_col), digits=6)), $(round(maximum(me_col), digits=6))]")
        end
    end
    
    return results
end

# Example: marginal effects for policy model
marginal_results = marginal_effects_analysis(model, data, [:age, :education_years, :experience])
```

## High-Frequency Trading Model

Real-time prediction serving with microsecond latency requirements:

```julia
using Dates

function high_frequency_trading_example()
    # Simulate high-frequency financial data
    Random.seed!(789)
    n_ticks = 100_000
    
    # Generate realistic financial time series
    returns = cumsum(randn(n_ticks) * 0.001)  # Random walk returns
    
    df = DataFrame(
        # Price features
        return_1min = returns,
        return_5min = lag(returns, 5),
        return_15min = lag(returns, 15),
        
        # Volume features  
        volume = abs.(randn(n_ticks)) .+ 1,
        volume_ratio = rand(0.5:0.01:2.0, n_ticks),
        
        # Market microstructure
        spread = abs.(randn(n_ticks)) * 0.01 .+ 0.001,
        market_impact = rand(0.001:0.0001:0.01, n_ticks),
        
        # Time features
        hour = repeat(9:16, inner=div(n_ticks, 8))[1:n_ticks],
        minute = repeat(0:59, inner=div(n_ticks, 60))[1:n_ticks],
        
        # Target: next minute return
        next_return = lead(returns, 1)
    )
    
    # Remove missing values from lags/leads
    df = df[16:(end-1), :]
    
    # Fit high-frequency prediction model
    model = lm(@formula(next_return ~ return_1min + return_5min + return_15min + 
                                    log(volume) + volume_ratio + spread + 
                                    market_impact + hour), df)
    
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    
    # Simulate real-time prediction serving
    println("High-Frequency Trading Model Performance:")
    println("Model coefficients: ", length(compiled))
    
    # Pre-allocate for zero-allocation serving
    row_vec = Vector{Float64}(undef, length(compiled))
    n_predictions = 10_000
    
    # Benchmark prediction latency
    prediction_times = Vector{Float64}(undef, n_predictions)
    predictions = Vector{Float64}(undef, n_predictions)
    
    println("Serving $n_predictions real-time predictions...")
    
    for i in 1:n_predictions
        tick_idx = rand(1:nrow(df))
        
        start_time = time_ns()
        compiled(row_vec, data, tick_idx)
        prediction = dot(coef(model), row_vec)
        end_time = time_ns()
        
        prediction_times[i] = (end_time - start_time) / 1000  # Convert to microseconds
        predictions[i] = prediction
    end
    
    # Latency analysis
    println("\nLatency Analysis:")
    println("Mean latency: $(round(mean(prediction_times), digits=2)) μs")
    println("Median latency: $(round(median(prediction_times), digits=2)) μs") 
    println("95th percentile: $(round(quantile(prediction_times, 0.95), digits=2)) μs")
    println("99th percentile: $(round(quantile(prediction_times, 0.99), digits=2)) μs")
    println("Max latency: $(round(maximum(prediction_times), digits=2)) μs")
    
    # Trading performance metrics
    actual_returns = [data.next_return[rand(1:nrow(df))] for _ in 1:n_predictions]
    
    # Simple trading strategy: long if predicted return > 0
    positions = sign.(predictions)
    strategy_returns = positions .* actual_returns
    
    println("\nTrading Strategy Performance:")
    println("Total predictions: $n_predictions")
    println("Accuracy (direction): $(round(100 * mean(sign.(predictions) .== sign.(actual_returns)), digits=1))%")
    println("Mean strategy return: $(round(mean(strategy_returns) * 10000, digits=2)) bps")
    println("Strategy Sharpe ratio: $(round(mean(strategy_returns) / std(strategy_returns), digits=3))")
    println("Max drawdown: $(round(minimum(cumsum(strategy_returns)) * 100, digits=2))%")
    
    return (latencies = prediction_times, predictions = predictions, returns = strategy_returns)
end

# Run high-frequency trading example
hft_results = high_frequency_trading_example()
```

## Medical Research: Clinical Trial Simulation

Simulating clinical trial outcomes with patient heterogeneity:

```julia
function clinical_trial_simulation()
    # Simulate diverse patient population
    Random.seed!(101112)
    n_patients = 5000
    
    df = DataFrame(
        # Patient demographics
        age = rand(18:85, n_patients),
        sex = categorical(rand(["Male", "Female"], n_patients)),
        bmi = max.(15, randn(n_patients) * 5 .+ 25),
        
        # Baseline health
        baseline_severity = rand(1:10, n_patients),
        comorbidities = rand(0:5, n_patients),
        
        # Treatment assignment (randomized)
        treatment = categorical(rand(["Placebo", "Low_Dose", "High_Dose"], n_patients)),
        
        # Compliance (realistic patterns)
        compliance_rate = min.(1.0, max.(0.0, randn(n_patients) * 0.2 .+ 0.8)),
        
        # Outcome: improvement score (0-100)
        improvement = max.(0, min.(100, 
            randn(n_patients) * 15 .+ 
            (df.treatment .== "High_Dose") * 20 .+
            (df.treatment .== "Low_Dose") * 10 .-
            df.baseline_severity * 2 .+
            df.compliance_rate * 15
        ))
    )
    
    # Clinical model
    model = lm(@formula(improvement ~ age + sex + bmi + baseline_severity + 
                                    comorbidities + treatment * compliance_rate), df)
    
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    
    # Define clinical scenarios
    scenarios = Dict(
        "real_world" => create_scenario("real_world", data),
        
        "perfect_compliance" => create_scenario("perfect_compliance", data;
            compliance_rate = 1.0
        ),
        
        "elderly_subgroup" => create_scenario("elderly", data;
            age = 70,  # All patients age 70
            compliance_rate = 0.9
        ),
        
        "high_risk_patients" => create_scenario("high_risk", data;
            baseline_severity = 8,
            comorbidities = 3,
            bmi = 30
        ),
        
        "optimal_candidates" => create_scenario("optimal", data;
            age = 45,
            baseline_severity = 5,
            comorbidities = 1,
            bmi = 23,
            compliance_rate = 1.0
        )
    )
    
    # Analyze treatment effects across scenarios
    println("Clinical Trial Scenario Analysis:")
    println("="^50)
    
    row_vec = Vector{Float64}(undef, length(compiled))
    
    for (scenario_name, scenario) in scenarios
        println("\nScenario: $scenario_name")
        
        # Calculate outcomes by treatment group
        treatment_groups = ["Placebo", "Low_Dose", "High_Dose"]
        group_results = Dict{String, Float64}()
        
        for treatment_group in treatment_groups
            # Create scenario with specific treatment
            treatment_scenario = create_scenario("temp", scenario.data;
                treatment = treatment_group
            )
            
            # Predict outcomes for all patients under this treatment
            outcomes = Vector{Float64}(undef, n_patients)
            for i in 1:n_patients
                compiled(row_vec, treatment_scenario.data, i)
                outcomes[i] = dot(coef(model), row_vec)
            end
            
            group_results[treatment_group] = mean(outcomes)
        end
        
        # Calculate treatment effects
        placebo_effect = group_results["Placebo"]
        low_dose_effect = group_results["Low_Dose"] - placebo_effect
        high_dose_effect = group_results["High_Dose"] - placebo_effect
        dose_response = high_dose_effect - low_dose_effect
        
        println("  Placebo response: $(round(placebo_effect, digits=1))")
        println("  Low dose effect: $(round(low_dose_effect, digits=1)) (vs placebo)")
        println("  High dose effect: $(round(high_dose_effect, digits=1)) (vs placebo)")
        println("  Dose response: $(round(dose_response, digits=1))")
        
        # Calculate number needed to treat (simplified)
        if high_dose_effect > 0
            nnt = round(Int, 100 / high_dose_effect)  # Assume 100-point scale
            println("  Number needed to treat: $nnt")
        end
    end
    
    return scenarios, group_results
end

# Run clinical trial analysis
clinical_scenarios, clinical_results = clinical_trial_simulation()
```

These examples demonstrate FormulaCompiler.jl's versatility across:

- **High-performance computing**: Monte Carlo simulations with millions of evaluations
- **Statistical inference**: Bootstrap confidence intervals with zero-allocation resampling  
- **Policy analysis**: Complex scenario modeling for decision support
- **Real-time systems**: Microsecond-latency prediction serving
- **Medical research**: Clinical trial simulation and subgroup analysis

Each example leverages FormulaCompiler.jl's core strengths: zero-allocation performance, flexible scenario system, and seamless integration with the Julia statistical ecosystem.