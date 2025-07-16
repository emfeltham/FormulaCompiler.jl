# test/test_mixed_models.jl
# Tests for MixedModels.jl integration

@testset "Mixed Models Integration" begin
    
    Random.seed!(42)
    
    # Create hierarchical data
    n_groups = 10
    n_per_group = 20
    n_total = n_groups * n_per_group
    
    df = DataFrame(
        subject = repeat(1:n_groups, inner=n_per_group),
        x = randn(n_total),
        z = randn(n_total),
        group = categorical(repeat(["A", "B"], inner=n_per_group÷2, outer=n_groups)),
        y = randn(n_total)
    )
    
    # Add group-level effects
    group_effects = randn(n_groups)
    df.y += group_effects[df.subject]
    
    @testset "Linear Mixed Models" begin
        # Test simple random intercept model
        model = fit(MixedModel, @formula(y ~ x + (1|subject)), df)
        compiled = compile_formula(model)
        
        # Test that fixed effects are extracted correctly
        fixed_form = fixed_effects_form(model)
        @test fixed_form.rhs isa AbstractTerm
        @test occursin("x", string(fixed_form.rhs))
        @test !occursin("|", string(fixed_form.rhs))  # No random effects
        
        # Test evaluation
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        
        # Should match fixed effects model matrix
        fixed_mm = modelmatrix(model)[1, :]
        @test isapprox(row_vec, fixed_mm, rtol=1e-12)
        
        # Test that compilation works for multiple rows
        for i in 1:min(10, nrow(df))
            compiled(row_vec, data, i)
            expected = modelmatrix(model)[i, :]
            @test isapprox(row_vec, expected, rtol=1e-12)
        end
    end
    
    @testset "Random Slopes Model" begin
        # Test random slopes model
        model = fit(MixedModel, @formula(y ~ x + z + (1 + x|subject)), df)
        compiled = compile_formula(model)
        
        # Test fixed effects extraction
        fixed_form = fixed_effects_form(model)
        @test occursin("x", string(fixed_form.rhs))
        @test occursin("z", string(fixed_form.rhs))
        @test !occursin("|", string(fixed_form.rhs))
        
        # Test evaluation matches fixed effects
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        
        fixed_mm = modelmatrix(model)[1, :]
        @test isapprox(row_vec, fixed_mm, rtol=1e-12)
    end
    
    @testset "Multiple Random Effects" begin
        # Add another grouping variable
        df.cluster = categorical(repeat(1:5, inner=n_total÷5))
        
        # Test model with multiple random effects
        model = fit(MixedModel, @formula(y ~ x + z + (1|subject) + (1|cluster)), df)
        compiled = compile_formula(model)
        
        # Test fixed effects extraction
        fixed_form = fixed_effects_form(model)
        @test occursin("x", string(fixed_form.rhs))
        @test occursin("z", string(fixed_form.rhs))
        @test !occursin("|", string(fixed_form.rhs))
        @test !occursin("subject", string(fixed_form.rhs))
        @test !occursin("cluster", string(fixed_form.rhs))
        
        # Test evaluation
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        
        fixed_mm = modelmatrix(model)[1, :]
        @test isapprox(row_vec, fixed_mm, rtol=1e-12)
    end
    
    @testset "Mixed Models with Interactions" begin
        # Test fixed effects interaction
        model = fit(MixedModel, @formula(y ~ x * group + (1|subject)), df)
        compiled = compile_formula(model)
        
        # Test that interaction is preserved in fixed effects
        fixed_form = fixed_effects_form(model)
        @test occursin("group", string(fixed_form.rhs))
        @test occursin("x", string(fixed_form.rhs))
        
        # Test evaluation
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        
        fixed_mm = modelmatrix(model)[1, :]
        @test isapprox(row_vec, fixed_mm, rtol=1e-12)
        
        # Test that interaction terms are computed correctly
        @test length(row_vec) > 3  # Should have interaction terms
    end
    
    @testset "Mixed Models with Functions" begin
        # Test function terms in fixed effects
        model = fit(MixedModel, @formula(y ~ log(abs(x) + 1) + sqrt(abs(z) + 1) + (1|subject)), df)
        compiled = compile_formula(model)
        
        # Test evaluation
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        
        fixed_mm = modelmatrix(model)[1, :]
        @test isapprox(row_vec, fixed_mm, rtol=1e-12)
        
        # Test that functions are evaluated correctly
        @test all(isfinite.(row_vec))
    end
    
    @testset "Edge Cases" begin
        # Test intercept-only random effects
        model = fit(MixedModel, @formula(y ~ x + (1|subject)), df)
        compiled = compile_formula(model)
        
        fixed_form = fixed_effects_form(model)
        @test !occursin("|", string(fixed_form))
        
        # Test no fixed effects (just random effects)
        model2 = fit(MixedModel, @formula(y ~ 1 + (1 + x|subject)), df)
        compiled2 = compile_formula(model2)
        
        fixed_form2 = fixed_effects_form(model2)
        @test string(fixed_form2.rhs) == "1"  # Just intercept
        
        # Test evaluation
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled2))
        compiled2(row_vec, data, 1)
        
        @test length(row_vec) == 1  # Just intercept
        @test row_vec[1] == 1.0
    end
    
    @testset "Generalized Linear Mixed Models" begin
        # Create binary outcome
        df.y_binary = df.y .> median(df.y)
        
        # Test GLMM
        model = fit(MixedModel, @formula(y_binary ~ x + z + (1|subject)), df, Binomial())
        compiled = compile_formula(model)
        
        # Test fixed effects extraction
        fixed_form = fixed_effects_form(model)
        @test occursin("x", string(fixed_form.rhs))
        @test occursin("z", string(fixed_form.rhs))
        @test !occursin("|", string(fixed_form.rhs))
        
        # Test evaluation
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        
        fixed_mm = modelmatrix(model)[1, :]
        @test isapprox(row_vec, fixed_mm, rtol=1e-12)
    end
    
    @testset "Performance with Mixed Models" begin
        # Test that mixed models compile efficiently
        model = fit(MixedModel, @formula(y ~ x * group + z + (1 + x|subject)), df)
        
        # Compilation should be fast
        compile_time = @elapsed compiled = compile_formula(model)
        @test compile_time < 1.0  # Should compile in under 1 second
        
        # Evaluation should be fast and zero-allocation
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        eval_time = @elapsed compiled(row_vec, data, 1)
        @test eval_time < 0.001  # Should evaluate in under 1ms
        
        allocs = @allocated compiled(row_vec, data, 1)
        @test allocs == 0  # Zero allocations
    end
    
    @testset "Consistency with GLM" begin
        # Test that fixed effects match equivalent GLM model
        
        # Fit mixed model
        mixed_model = fit(MixedModel, @formula(y ~ x + z + (1|group)), df)
        compiled_mixed = compile_formula(mixed_model)
        
        # Fit equivalent GLM model
        glm_model = lm(@formula(y ~ x + z), df)
        compiled_glm = compile_formula(glm_model)
        
        # Should have same structure
        @test length(compiled_mixed) == length(compiled_glm)
        
        # Should give same results
        data = Tables.columntable(df)
        row_vec_mixed = Vector{Float64}(undef, length(compiled_mixed))
        row_vec_glm = Vector{Float64}(undef, length(compiled_glm))
        
        compiled_mixed(row_vec_mixed, data, 1)
        compiled_glm(row_vec_glm, data, 1)
        
        @test isapprox(row_vec_mixed, row_vec_glm, rtol=1e-12)
    end
    
    @testset "Fixed Effects Helper Functions" begin
        # Test fixed_effects_form function directly
        
        # Test with regular GLM (should be identity)
        glm_model = lm(@formula(y ~ x + z), df)
        fixed_form = fixed_effects_form(glm_model)
        @test fixed_form == formula(glm_model)
        
        # Test with mixed model
        mixed_model = fit(MixedModel, @formula(y ~ x + z + (1|subject)), df)
        fixed_form = fixed_effects_form(mixed_model)
        
        @test fixed_form.lhs == formula(mixed_model).lhs
        @test !occursin("|", string(fixed_form.rhs))
        @test occursin("x", string(fixed_form.rhs))
        @test occursin("z", string(fixed_form.rhs))
        
        # Test with complex random effects
        complex_model = fit(MixedModel, @formula(y ~ x + z + (1 + x|subject) + (1|cluster)), df)
        fixed_form = fixed_effects_form(complex_model)
        
        @test !occursin("|", string(fixed_form.rhs))
        @test !occursin("subject", string(fixed_form.rhs))
        @test !occursin("cluster", string(fixed_form.rhs))
        @test occursin("x", string(fixed_form.rhs))
        @test occursin("z", string(fixed_form.rhs))
    end
    
end
