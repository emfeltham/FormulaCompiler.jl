# Test conditional logic and boolean negation in formulas
using DataFrames, GLM, FormulaCompiler, Tables
using Test

@testset "Formula Logic Tests" begin
    
    # Create test data with proper categorical handling
    df = DataFrame(
        y = randn(100),
        x = randn(100),
        group = rand(["A", "B"], 100),
        flag = rand(Bool, 100)
    )
    
    # Convert group to categorical for proper handling
    using CategoricalArrays
    df.group = categorical(df.group)
    
    @testset "Conditional Logic in Formulas" begin
        # Test if FormulaCompiler can handle conditional logic like (x <= 2.0)
        model = lm(@formula(y ~ x + (x <= 2.0)), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        
        # Check correctness against reference
        ref_mm = modelmatrix(model)
        output = Vector{Float64}(undef, length(compiled))
        
        for i in 1:min(10, size(ref_mm, 1))  # Test first 10 rows
            compiled(output, data, i)
            @test output ≈ ref_mm[i, :] atol=1e-10
        end
    end
    
    @testset "Boolean Negation in Formulas" begin
        # Test if FormulaCompiler can handle boolean negation like !flag
        model = lm(@formula(y ~ x + !flag), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        
        # Check correctness against reference
        ref_mm = modelmatrix(model)
        output = Vector{Float64}(undef, length(compiled))
        
        for i in 1:min(10, size(ref_mm, 1))  # Test first 10 rows
            compiled(output, data, i)
            @test output ≈ ref_mm[i, :] atol=1e-10
        end
    end
    
    @testset "Complex Logic Combinations" begin
        # Test combination of issues from tough_formula.md
        df.close_dist = df.x .<= 2.0
        df.not_flag = .!df.flag
        
        # This should work with preprocessed variables and verify correctness
        model = lm(@formula(y ~ x + close_dist & group + not_flag & x), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        
        # Check correctness against reference
        ref_mm = modelmatrix(model)
        output = Vector{Float64}(undef, length(compiled))
        
        for i in 1:min(10, size(ref_mm, 1))  # Test first 10 rows
            compiled(output, data, i)
            @test output ≈ ref_mm[i, :] atol=1e-10
        end
    end    
end
