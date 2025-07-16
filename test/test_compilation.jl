# test/test_compilation.jl
# Tests for formula compilation and code generation

@testset "Formula Compilation" begin
    
    # Create comprehensive test data
    Random.seed!(42)
    df = DataFrame(
        x = randn(100),
        y = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100)),
        flag = rand([true, false], 100),
        cat2 = categorical(rand(["X", "Y"], 100))
    )
    data = Tables.columntable(df)
    
    @testset "Basic Formula Compilation" begin
        # Simple linear model
        model = lm(@formula(y ~ x), df)
        compiled = compile_formula(model)
        
        @test compiled isa CompiledFormula
        @test compiled.output_width == 2  # intercept + x
        @test :x in compiled.column_names
        @test compiled.root_evaluator isa AbstractEvaluator
        
        # Test evaluation
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Categorical Variables" begin
        # Test categorical with default contrasts
        model = lm(@formula(y ~ group), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test binary categorical
        model = lm(@formula(y ~ cat2), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Function Terms" begin
        # Test mathematical functions
        model = lm(@formula(y ~ log(z)), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test polynomial terms
        model = lm(@formula(y ~ x^2), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Interactions" begin
        # Test continuous × continuous
        model = lm(@formula(y ~ x * z), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test continuous × categorical
        model = lm(@formula(y ~ x * group), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test categorical × categorical
        model = lm(@formula(y ~ group * cat2), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Complex Formulas" begin
        # Test three-way interaction
        model = lm(@formula(y ~ x * z * group), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test function in interaction
        model = lm(@formula(y ~ log(z) * group), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test boolean function
        model = lm(@formula(y ~ (x > 0) * group), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Edge Cases" begin
        # Test intercept-only model
        model = lm(@formula(y ~ 1), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test no-intercept model
        model = lm(@formula(y ~ 0 + x), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Compilation Caching" begin
        # Test that compilation is cached
        model = lm(@formula(y ~ x * group), df)
        
        # First compilation
        compiled1 = compile_formula(model)
        
        # Second compilation should use cache
        compiled2 = compile_formula(model)
        
        # Should produce same results
        row_vec1 = Vector{Float64}(undef, length(compiled1))
        row_vec2 = Vector{Float64}(undef, length(compiled2))
        
        compiled1(row_vec1, data, 1)
        compiled2(row_vec2, data, 1)
        
        @test row_vec1 == row_vec2
    end
    
end
