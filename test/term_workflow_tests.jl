# test_modeling_workflow.jl - Real modeling workflow tests

# Helper function for logistic
invlogit(x) = 1 / (1 + exp(-x))

@testset "Real Modeling Workflow Tests" begin
    @testset "Simple Linear Model Workflow" begin
        # Generate synthetic data
        Random.seed!(123)
        n = 100
        df = DataFrame(
            x = randn(n),
            z = randn(n),
            group = categorical(rand(["A", "B", "C"], n)),
            y = zeros(n)
        )
        
        # Create response with known relationship
        df.y = 2.0 .+ 1.5 .* df.x .- 0.8 .* df.z .+ 
               (df.group .== "B") .* 0.5 .+ (df.group .== "C") .* (-0.3) .+ 
               0.1 .* randn(n)
        
        # Fit model: y ~ 1 + x + z + group
        formula_obj = @formula(y ~ 1 + x + z + group)
        model = lm(formula_obj, df)
        
        @test typeof(model) <: StatisticalModel
        
        # Extract RHS and build column mapping
        rhs = fixed_effects_form(model).rhs
        mapping = build_column_mapping(rhs)
        
        # Test basic structure
        @test mapping.total_columns == 5  # intercept + x + z + group(2 dummies)
        
        # Test variable mappings
        @test haskey(mapping.symbol_to_ranges, :x)
        @test haskey(mapping.symbol_to_ranges, :z) 
        @test haskey(mapping.symbol_to_ranges, :group)
        
        # x and z should each have 1 column
        @test length(get_all_variable_columns(mapping, :x)) == 1
        @test length(get_all_variable_columns(mapping, :z)) == 1
        
        # group should have 2 columns (3 levels - 1 reference)
        @test length(get_all_variable_columns(mapping, :group)) == 2
        
        # Test InplaceModeler integration
        ipm = InplaceModeler(model, nrow(df))
        @test ipm isa InplaceModeler
        
        # Build model matrix using both approaches
        X_original = modelmatrix(model)
        X_efficient = Matrix{Float64}(undef, nrow(df), size(X_original, 2))
        
        data_tbl = Tables.columntable(df)
        modelmatrix!(ipm, data_tbl, X_efficient)
        
        # Should be identical (within floating point precision)
        @test X_original ≈ X_efficient
        
        # Test variable extraction using mapping
        ipm_mapping = InplaceModelerWithMapping(model, nrow(df))
        x_cols = get_variable_ranges(ipm_mapping.mapping, :x)
        z_cols = get_variable_ranges(ipm_mapping.mapping, :z)
        group_cols = get_variable_ranges(ipm_mapping.mapping, :group)
        
        @test length(x_cols) == 1    # x appears once
        @test length(z_cols) == 1    # z appears once  
        @test length(group_cols) == 1  # group appears once (but spans 2 columns)
        
        # Extract specific variable columns
        X_x = X_efficient[:, x_cols[1]]
        X_z = X_efficient[:, z_cols[1]]
        X_group = X_efficient[:, group_cols[1]]
        
        @test X_x ≈ df.x  # Should match original x values
        @test X_z ≈ df.z  # Should match original z values
        @test size(X_group, 2) == 2  # group should have 2 dummy columns
    end

    @testset "Complex Formula with Function Terms" begin
        # Simulate data for: y ~ x + x^2 + log(abs(x) + 1) + z + x&z
        Random.seed!(456)
        n = 200
        df = DataFrame(
            x = 2 .* randn(n),
            z = randn(n)
        )
        
        # Create complex response
        df.y = 1.0 .+ 
               0.8 .* df.x .+                           # linear x
               -0.2 .* (df.x .^ 2) .+                   # quadratic x
               0.5 .* log.(abs.(df.x) .+ 1) .+          # log transform
               1.2 .* df.z .+                           # linear z  
               0.3 .* (df.x .* df.z) .+                 # interaction
               0.15 .* randn(n)                         # noise
        
        # Create the complex formula
        # Note: Using identity() as a proxy for x^2 since ^ might not work directly in @formula
        formula_obj = @formula(y ~ x + x^2 + log(abs(x) + 1) + z + x&z)
        model = lm(formula_obj, df)
        
        @test model isa StatisticalModel
        
        # Extract and analyze the model structure
        rhs = fixed_effects_form(model).rhs
        mapping = build_column_mapping(rhs)
        
        # Should have multiple columns due to function terms and interactions
        @test mapping.total_columns >= 5  # At least: intercept, x, I(x^2), log(...), z, x&z
        
        # x should appear in multiple terms: x, I(x^2), log(...), x&z
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) >= 3  # x appears in multiple contexts
        
        # z should appear in: z, x&z
        z_ranges = get_variable_ranges(mapping, :z)
        @test length(z_ranges) >= 2  # z appears in multiple contexts
        
        # Test detailed analysis
        analysis = analyze_formula_structure(mapping)
        @test haskey(analysis, :x)
        @test haskey(analysis, :z)
        
        @test analysis[:x]["appears_in_terms"] >= 3
        @test analysis[:z]["appears_in_terms"] >= 2
        
        # Test InplaceModeler works with complex formula
        ipm = InplaceModeler(model, nrow(df))
        X_original = modelmatrix(model)
        X_efficient = Matrix{Float64}(undef, nrow(df), size(X_original, 2))
        
        data_tbl = Tables.columntable(df)
        modelmatrix!(ipm, data_tbl, X_efficient)
        
        @test X_original ≈ X_efficient
        
        # Test that we can identify all columns involving x
        ipm_mapping = InplaceModelerWithMapping(model, nrow(df))
        all_x_cols = get_all_variable_columns(ipm_mapping.mapping, :x)
        all_z_cols = get_all_variable_columns(ipm_mapping.mapping, :z)
        
        @test length(all_x_cols) >= 3  # x participates in multiple columns
        @test length(all_z_cols) >= 2  # z participates in multiple columns
        
        # Verify we can get terms involving each variable
        x_terms = get_terms_involving_variable(ipm_mapping.mapping, :x)
        z_terms = get_terms_involving_variable(ipm_mapping.mapping, :z)
        
        @test length(x_terms) >= 3
        @test length(z_terms) >= 2
    end

    @testset "GLM with Categorical Interactions" begin
        # Test with logistic regression and categorical interactions
        Random.seed!(789)
        n = 300
        df = DataFrame(
            x = randn(n),
            treatment = categorical(rand(["Control", "Drug A", "Drug B"], n)),
            age_group = categorical(rand(["Young", "Old"], n)),
        )
        
        # Create binary response with interactions
        logit = -1.0 .+ 
                0.5 .* df.x .+
                (df.treatment .== "Drug A") .* 0.8 .+
                (df.treatment .== "Drug B") .* 1.2 .+
                (df.age_group .== "Old") .* 0.6 .+
                (df.treatment .== "Drug A") .* (df.age_group .== "Old") .* 0.4 .+
                (df.treatment .== "Drug B") .* (df.age_group .== "Old") .* (-0.3)
        
        df.y = rand.(Bernoulli.(invlogit.(logit)))
        
        # Fit logistic model with interactions
        formula_obj = @formula(y ~ x + treatment * age_group)
        model = glm(formula_obj, df, Binomial(), LogitLink())
        
        @test model isa StatisticalModel
        
        # Extract structure
        rhs = fixed_effects_form(model).rhs  
        mapping = build_column_mapping(rhs)
        
        # Should have: intercept + x + treatment(2) + age_group(1) + treatment:age_group(2)
        @test mapping.total_columns >= 6
        
        # All variables should be detected
        @test haskey(mapping.symbol_to_ranges, :x)
        @test haskey(mapping.symbol_to_ranges, :treatment)  
        @test haskey(mapping.symbol_to_ranges, :age_group)
        
        # treatment and age_group should appear in main effects AND interactions
        treatment_ranges = get_variable_ranges(mapping, :treatment)
        age_group_ranges = get_variable_ranges(mapping, :age_group)
        
        @test length(treatment_ranges) >= 2  # main effect + interaction
        @test length(age_group_ranges) >= 2  # main effect + interaction
        
        # x should only appear once (no interactions with x)
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) == 1
        
        # Test InplaceModeler with GLM
        ipm = InplaceModeler(model, nrow(df))
        X_original = modelmatrix(model)
        X_efficient = Matrix{Float64}(undef, nrow(df), size(X_original, 2))
        
        data_tbl = Tables.columntable(df)
        modelmatrix!(ipm, data_tbl, X_efficient)
        
        @test X_original ≈ X_efficient
        
        # Test variable column extraction
        ipm_mapping = InplaceModelerWithMapping(model, nrow(df))
        
        treatment_cols = get_all_variable_columns(ipm_mapping.mapping, :treatment)
        age_group_cols = get_all_variable_columns(ipm_mapping.mapping, :age_group)
        
        # treatment appears in main effect (2 cols) + interaction (2 cols) = 4 total
        # age_group appears in main effect (1 col) + interaction (2 cols) = 3 total  
        @test length(treatment_cols) >= 2
        @test length(age_group_cols) >= 2
    end

    @testset "Nested Functions and Complex Transformations" begin
        # Test very complex formula: y ~ poly(x, 2) + log(z + abs(w)) + group + poly(x,2):group
        Random.seed!(101112)
        n = 150
        df = DataFrame(
            x = 3 .* randn(n),
            z = 2 .* randn(n), 
            w = randn(n),
            group = categorical(rand(["A", "B"], n))
        )
        
        # Create response with polynomial and complex transformations
        df.y = 2.0 .+
               1.2 .* df.x .+ (-0.3) .* (df.x .^ 2) .+           # polynomial in x
               0.8 .* inv.(df.z .+ abs.(df.w) .+ 0.1) .+         # complex log transform
               (df.group .== "B") .* 1.5 .+                      # group effect
               (df.group .== "B") .* (0.4 .* df.x .+ (-0.1) .* (df.x .^ 2)) .+  # group:poly interaction
               0.2 .* randn(n)
        
        # Note: Using simpler transformations that work with @formula macro
        formula_obj = @formula(y ~ x + x^2 + inv(z + abs(w) + 0.1) + group + (x + x^2)&group)
        model = lm(formula_obj, df)
        
        @test model isa StatisticalModel
        
        # Extract and analyze
        rhs = fixed_effects_form(model).rhs
        mapping = build_column_mapping(rhs)
        
        # Complex formula should generate many columns
        @test mapping.total_columns >= 7
        
        # x should appear in many contexts: x, I(x^2), interactions
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) >= 3
        
        # All variables should be detected
        @test haskey(mapping.symbol_to_ranges, :x)
        @test haskey(mapping.symbol_to_ranges, :z)
        @test haskey(mapping.symbol_to_ranges, :w) 
        @test haskey(mapping.symbol_to_ranges, :group)
        
        # z and w should appear in the log(...) term
        z_ranges = get_variable_ranges(mapping, :z)
        w_ranges = get_variable_ranges(mapping, :w)
        @test length(z_ranges) >= 1
        @test length(w_ranges) >= 1
        
        # group should appear in main effect and interactions
        group_ranges = get_variable_ranges(mapping, :group)
        @test length(group_ranges) >= 2
        
        # Test comprehensive analysis
        analysis = analyze_formula_structure(mapping)
        @test analysis[:x]["appears_in_terms"] >= 3
        @test analysis[:group]["appears_in_terms"] >= 2
        
        # Test InplaceModeler still works
        ipm = InplaceModeler(model, nrow(df))
        X_original = modelmatrix(model)
        X_efficient = Matrix{Float64}(undef, nrow(df), size(X_original, 2))
        
        data_tbl = Tables.columntable(df)
        modelmatrix!(ipm, data_tbl, X_efficient)
        
        @test X_original ≈ X_efficient
        
        # Test that mapping correctly identifies all variable participations
        ipm_mapping = InplaceModelerWithMapping(model, nrow(df))
        
        for var in [:x, :z, :w, :group]
            cols = get_all_variable_columns(ipm_mapping.mapping, var)
            @test length(cols) >= 1
            @test all(1 ≤ col ≤ mapping.total_columns for col in cols)
        end
    end

    @testset "Real-world Margins Use Case" begin
        # Demonstrate how this would be used in practice for margins/AME computation
        Random.seed!(131415)
        n = 100
        df = DataFrame(
            x = randn(n),
            z = randn(n),
            group = categorical(rand(["Low", "High"], n))
        )
        
        # Response with interaction
        df.y = 1.0 .+ 0.8 .* df.x .+ 0.5 .* df.z .+ 
               (df.group .== "High") .* 0.6 .+
               (df.group .== "High") .* 0.4 .* df.x .+  # interaction
               0.1 .* randn(n)
        
        formula_obj = @formula(y ~ x + z + group + x&group)
        model = lm(formula_obj, df)
        
        # This is what your margins code would do:
        ipm_mapping = InplaceModelerWithMapping(model, nrow(df))
        
        # For computing marginal effects of x:
        # 1. Identify all columns that depend on x
        x_columns = get_all_variable_columns(ipm_mapping.mapping, :x)
        @test length(x_columns) >= 2  # x main effect + x:group interaction
        
        # 2. Build base model matrix
        X_base = Matrix{Float64}(undef, nrow(df), ipm_mapping.mapping.total_columns)
        data_tbl = Tables.columntable(df)
        modelmatrix!(ipm_mapping.modeler, data_tbl, X_base)
        
        # 3. For derivative computation, only these columns would need to be recomputed:
        affected_ranges = get_variable_ranges(ipm_mapping.mapping, :x)
        @test length(affected_ranges) >= 2
        
        # 4. Get detailed term information for sophisticated derivative computation
        x_term_info = get_terms_involving_variable(ipm_mapping.mapping, :x)
        @test length(x_term_info) >= 2
        
        # Verify each term info is reasonable
        for (term, range) in x_term_info
            @test term isa AbstractTerm
            @test 1 ≤ range.start ≤ range.stop ≤ ipm_mapping.mapping.total_columns
        end
        
        # 5. This enables efficient computation: only recompute affected columns
        # rather than rebuilding the entire model matrix
        total_affected_cols = length(x_columns)
        total_cols = ipm_mapping.mapping.total_columns
        efficiency_ratio = 1.0 - (total_affected_cols / total_cols)
        
        @test efficiency_ratio > 0  # We should save some computation
        @test total_affected_cols < total_cols  # Not all columns depend on x
        
        println("Efficiency test: x affects $total_affected_cols out of $total_cols columns")
        println("Potential computation savings: $(round(efficiency_ratio * 100, digits=1))%")
    end
end

