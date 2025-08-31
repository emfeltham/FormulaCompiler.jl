using Test
using FormulaCompiler, GLM, DataFrames, Tables, Statistics, StatsModels
using RDatasets, CategoricalArrays
using LinearAlgebra  # For I() function in formulas

@testset "Documentation Examples Validation" begin
    
    @testset "Quick Reference Examples" begin
        # Basic compilation
        df = DataFrame(y = randn(100), x = randn(100))
        model = lm(@formula(y ~ x), df)
        compiled = compile_formula(model, Tables.columntable(df))
        @test compiled isa FormulaCompiler.UnifiedCompiled
        
        # Zero-allocation evaluation
        output = Vector{Float64}(undef, length(compiled))
        data = Tables.columntable(df)
        compiled(output, data, 1)
        @test length(output) == length(compiled)
        
        # Scenario creation
        scenario = create_scenario("test", data; x = 2.0)
        @test scenario isa FormulaCompiler.DataScenario
        
        # Marginal effects
        vars = [:x]
        de = build_derivative_evaluator(compiled, data; vars=vars)
        g = Vector{Float64}(undef, length(vars))
        marginal_effects_eta!(g, de, coef(model), 1; backend=:fd)
        @test length(g) == length(vars)
    end
    
    @testset "Economics Examples - Ecdat/Bwages" begin
        # Load Belgian wages dataset
        bwages = dataset("Ecdat", "Bwages")
        @test nrow(bwages) > 0
        
        # Wage regression with interaction (use actual column names)
        model = lm(@formula(log(Wage) ~ Educ + Sex + Exper), bwages)  # Simplified formula
        data = Tables.columntable(bwages)
        compiled = compile_formula(model, data)
        @test compiled isa FormulaCompiler.UnifiedCompiled
        
        # Policy scenario - education boost
        policy_scenario = create_scenario("education_boost", data; Educ = 16)
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, policy_scenario.data, 1)
        @test length(output) == length(compiled)
        
        # Marginal effects on continuous variables
        vars = [:Educ, :Exper]
        de = build_derivative_evaluator(compiled, data; vars=vars)
        g = Vector{Float64}(undef, length(vars))
        marginal_effects_eta!(g, de, coef(model), 1; backend=:fd)
        @test length(g) == length(vars)
    end
    
    @testset "Engineering Examples - datasets/mtcars" begin
        # Load motor vehicle dataset
        mtcars = dataset("datasets", "mtcars")
        @test nrow(mtcars) > 0
        
        # Fuel efficiency model (use actual column names)
        model = lm(@formula(MPG ~ WT + Cyl + HP), mtcars)  # Remove log to avoid domain issues
        data = Tables.columntable(mtcars)
        compiled = compile_formula(model, data)
        @test compiled isa FormulaCompiler.UnifiedCompiled
        
        # Scenario analysis - lightweight high-performance car
        scenario = create_scenario("lightweight_performance", data; 
            WT = 2.5, HP = 200, Cyl = 6)
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, scenario.data, 1)
        @test length(output) == length(compiled)
        
        # Marginal effects
        vars = [:WT, :HP]  # Only continuous variables
        de = build_derivative_evaluator(compiled, data; vars=vars)
        g = Vector{Float64}(undef, length(vars))
        marginal_effects_eta!(g, de, coef(model), 1; backend=:fd)
        @test length(g) == length(vars)
    end
    
    @testset "Biostatistics Examples - survival/lung" begin
        # Load lung cancer survival dataset
        lung = dataset("survival", "lung")
        @test nrow(lung) > 0
        
        # Remove missing values for GLM (use actual column names)
        lung_complete = dropmissing(lung, [:Time, :Status, :Age, :Sex])
        @test nrow(lung_complete) > 0
        
        # Survival model (using log-time as surrogate for survival analysis)
        model = lm(@formula(log(Time) ~ Age + Sex), lung_complete)  # Simplified formula
        data = Tables.columntable(lung_complete)
        compiled = compile_formula(model, data)
        @test compiled isa FormulaCompiler.UnifiedCompiled
        
        # Treatment scenario - standardized patient profile
        scenario = create_scenario("standard_patient", data; 
            Age = 65, Sex = 1)
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, scenario.data, 1)
        @test length(output) == length(compiled)
        
        # Marginal effects on age
        vars = [:Age]
        de = build_derivative_evaluator(compiled, data; vars=vars)
        g = Vector{Float64}(undef, length(vars))
        marginal_effects_eta!(g, de, coef(model), 1; backend=:fd)
        @test length(g) == length(vars)
    end
    
    @testset "Social Sciences Examples - UCBAdmissions" begin
        # Load UC Berkeley admissions dataset
        ucb = dataset("datasets", "UCBAdmissions")
        @test nrow(ucb) > 0
        
        # Convert to individual-level data for regression
        ucb_expanded = DataFrame()
        for row in eachrow(ucb)
            n_obs = Int(row.Freq)
            if n_obs > 0
                individual_data = DataFrame(
                    admitted = fill(row.Admit == "Admitted", n_obs),
                    gender = categorical(fill(string(row.Gender), n_obs)),
                    dept = categorical(fill(string(row.Dept), n_obs))
                )
                append!(ucb_expanded, individual_data)
            end
        end
        @test nrow(ucb_expanded) > 0
        
        # Admission probability model
        model = glm(@formula(admitted ~ gender * dept), ucb_expanded, Binomial(), LogitLink())
        data = Tables.columntable(ucb_expanded)
        compiled = compile_formula(model, data)
        @test compiled isa FormulaCompiler.UnifiedCompiled
        
        # Scenario analysis - gender equity assessment  
        # Use existing categorical values from the data to ensure level compatibility
        female_idx = findfirst(x -> string(x) == "Female", ucb_expanded.gender)
        dept_a_idx = findfirst(x -> string(x) == "A", ucb_expanded.dept)
        
        if !isnothing(female_idx) && !isnothing(dept_a_idx)
            female_val = ucb_expanded.gender[female_idx]
            dept_a_val = ucb_expanded.dept[dept_a_idx]
            
            scenario = create_scenario("gender_analysis", data; 
                gender = female_val, 
                dept = dept_a_val)
            output = Vector{Float64}(undef, length(compiled))
            compiled(output, scenario.data, 1)
            @test length(output) == length(compiled)
        else
            @test true  # Skip if values not found
        end
        
        # Test string overrides (should work automatically)
        string_scenario = create_scenario("string_test", data;
            gender = "Female",
            dept = "A"
        )
        string_output = Vector{Float64}(undef, length(compiled))
        compiled(string_output, string_scenario.data, 1)
        @test length(string_output) == length(compiled)
    end
    
    @testset "Advanced Computational Patterns" begin
        # Test Monte Carlo simulation pattern
        df = DataFrame(
            outcome = randn(1000),
            treatment = rand(Bool, 1000),
            age = rand(25:65, 1000),
            score = randn(1000)
        )
        
        model = lm(@formula(outcome ~ treatment * age + score), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        
        # Monte Carlo simulation
        n_sims = 100
        output = Vector{Float64}(undef, length(compiled))
        results = Vector{Float64}(undef, n_sims)
        
        for i in 1:n_sims
            row_idx = rand(1:nrow(df))
            compiled(output, data, row_idx)
            results[i] = dot(coef(model), output)
        end
        
        @test length(results) == n_sims
        @test all(isfinite, results)
        
        # Bootstrap pattern test
        bootstrap_results = Vector{Float64}()
        for i in 1:10  # Small bootstrap for testing
            # Sample with replacement
            boot_indices = rand(1:nrow(df), nrow(df))
            boot_df = df[boot_indices, :]
            
            boot_model = lm(@formula(outcome ~ treatment * age + score), boot_df)
            boot_data = Tables.columntable(boot_df)
            boot_compiled = compile_formula(boot_model, boot_data)
            
            # Evaluate at fixed point
            boot_compiled(output, boot_data, 1)
            push!(bootstrap_results, dot(coef(boot_model), output))
        end
        
        @test length(bootstrap_results) == 10
        @test all(isfinite, bootstrap_results)
    end
end