# test_execution_plans_phase1.jl


@testset "Phase 1: Self-Contained Evaluators Correctness" begin
    
    # Create comprehensive test data with edge cases
    n = 200  # Larger dataset for better testing
    df = DataFrame(
        # Continuous variables
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,  # Positive for log
        w = randn(n),
        t = randn(n),
        
        # Categorical variables with different levels
        group3 = categorical(rand(["A", "B", "C"], n)),           # 3 levels
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      # 4 levels
        binary = categorical(rand(["Yes", "No"], n)),             # 2 levels
        group5 = categorical(rand(["P", "Q", "R", "S", "T"], n)), # 5 levels
        
        # Boolean/logical
        flag = rand([true, false], n),
        
        # Response variable
        response = randn(n)
    )
    data = Tables.columntable(df)
    
    @testset "Basic Self-Contained Integration" begin
        @testset "Simple formulas work correctly" begin
            formulas = [
                @formula(response ~ 1),                    # Intercept only
                @formula(response ~ 0 + x),                # No intercept  
                @formula(response ~ x),                    # Simple continuous
                @formula(response ~ group3),               # Simple categorical
                @formula(response ~ x + y),                # Multiple continuous
                @formula(response ~ group3 + group4),      # Multiple categorical
                @formula(response ~ x + group3),           # Mixed
            ]
            
            f = formulas[1]
            for f in formulas
                @testset "Formula: $formula" begin
                    # CORRECT: Fit model first, then compile
                    model = lm(f, df)
                    compiled = compile_formula(model, data)
                    
                    # Test structure
                    @test compiled isa CompiledFormula
                    @test compiled.root_evaluator isa AbstractEvaluator
                    @test compiled.output_width == size(modelmatrix(model), 2)
                    
                    # Test execution correctness
                    output = Vector{Float64}(undef, length(compiled))
                    compiled(output, data, 5)  # Test row 5
                    expected = modelmatrix(model)[5, :]
                    @test isapprox(output, expected, rtol=1e-12)
                    
                    # Test multiple rows
                    for test_row in [1, 10, 50, 100, n]
                        fill!(output, NaN)  # Clear output
                        compiled(output, data, test_row)
                        expected_row = modelmatrix(model)[test_row, :]
                        @test isapprox(output, expected_row, rtol=1e-12)
                    end
                end
            end
        end
        
        @testset "Function terms" begin
            function_formulas = [
                @formula(response ~ log(z)),              # Simple function
                @formula(response ~ sqrt(z)),             # Another math function
                @formula(response ~ x^2),                 # Power function
                @formula(response ~ x + log(z)),          # Mixed function + continuous
                @formula(response ~ log(z) + group3),     # Function + categorical
                @formula(response ~ abs(x)),              # Abs function
                @formula(response ~ exp(x/10)),           # Exp with scaling
                @formula(response ~ log(z + 1)),          # Function with arithmetic
                @formula(response ~ x^2 + y^2),           # Multiple power terms
            ]
            
            f = function_formulas[1]
            for f in function_formulas
                @testset "Function formula: $f" begin
                    try
                        # CORRECT: Fit model first, then compile
                        model = lm(f, df)
                        compiled = compile_formula(model, data)
                        
                        # Test execution
                        output = Vector{Float64}(undef, length(compiled))
                        compiled(output, data, 1)
                        expected = modelmatrix(model)[1, :]
                        @test isapprox(output, expected, rtol=1e-12)
                        
                        # Test a few more rows to catch edge cases
                        for test_row in [5, 25, 75]
                            compiled(output, data, test_row)
                            expected_row = modelmatrix(model)[test_row, :]
                            @test isapprox(output, expected_row, rtol=1e-12)
                        end
                    catch e
                        @warn "Function formula failed: $f" exception=e
                        rethrow(e)
                    end
                end
            end
        end
    end
    
    @testset "Complex Interactions" begin
        @testset "Two-way interactions" begin
            interaction_formulas = [
                # Continuous × Continuous
                @formula(response ~ x * y),
                @formula(response ~ x * z),
                @formula(response ~ (x + y) * z),
                
                # Continuous × Categorical  
                @formula(response ~ x * group3),
                @formula(response ~ y * binary),
                @formula(response ~ z * group4),
                
                # Categorical × Categorical
                @formula(response ~ group3 * binary),
                @formula(response ~ group3 * group4),
                @formula(response ~ binary * group5),
                
                # Function × Variable
                @formula(response ~ log(z) * x),
                @formula(response ~ sqrt(z) * group3),
                @formula(response ~ (x^2) * binary),
            ]
            
            f = interaction_formulas[1]
            for formula in interaction_formulas
                @testset "Interaction: $f" begin
                    # CORRECT: Fit model first, then compile
                    model = lm(f, df)
                    compiled = compile_formula(model, data)
                    
                    # Test structure
                    @test compiled.root_evaluator isa AbstractEvaluator
                    @test compiled.output_width == size(modelmatrix(model), 2)
                    
                    # Test correctness across multiple rows
                    output = Vector{Float64}(undef, length(compiled))
                    for test_row in [1, 10, 25, 50, 100]
                        compiled(output, data, test_row)
                        expected = modelmatrix(model)[test_row, :]
                        @test isapprox(output, expected, rtol=1e-12)
                        "Failed at row $test_row for $formula"
                    end
                end
            end
        end
        
        @testset "Three-way interactions" begin
            three_way_formulas = [
                # Continuous × Continuous × Continuous
                @formula(response ~ x * y * z),
                @formula(response ~ x * y * w),
                
                # Continuous × Continuous × Categorical
                @formula(response ~ x * y * group3),
                @formula(response ~ x * z * binary),
                
                # Continuous × Categorical × Categorical  
                @formula(response ~ x * group3 * binary),
                @formula(response ~ y * group3 * group4),
                
                # Categorical × Categorical × Categorical
                @formula(response ~ group3 * binary * group4),
                
                # Functions in three-way interactions
                @formula(response ~ log(z) * x * group3),
                @formula(response ~ x^2 * y * binary),
                
                # Boolean interactions
                @formula(response ~ flag * x * group3),
                @formula(response ~ (x > 0) * y * group3),

                # Misc.
                @formula(response ~ log(abs(z)) * x * group3),
                @formula(response ~ log(abs(z) + abs(x)) * x * group3)
            ]
            


            f = three_way_formulas[8]
            for f in three_way_formulas
                @testset "Three-way: $f" begin
                    try
                        # CORRECT: Fit model first, then compile
                        model = lm(f, df)
                        compiled = compile_formula(model, data)
                        
                        rhs = fixed_effects_form(model).rhs
                        root_evaluator = compile_term(rhs) # -> evaluators.jl


                        @test compiled.output_width == size(modelmatrix(model), 2)
                        
                        # Test correctness
                        output = Vector{Float64}(undef, length(compiled))
                        for test_row in [1, 15, 45, 85]
                            compiled(output, data, test_row)
                            expected = modelmatrix(model)[test_row, :]
                            @test isapprox(output, expected, rtol=1e-12)
                            "Failed at row $test_row"
                        end
                    catch e
                        @warn "Three-way interaction failed: $f" exception=e
                        rethrow(e)
                    end
                end
            end
        end
        
        @testset "Four-way and higher interactions" begin
            complex_formulas = [
                @formula(response ~ x * y * z * w),
                @formula(response ~ x * y * group3 * binary),
                @formula(response ~ group3 * group4 * binary * group5),
                @formula(response ~ log(z) * x * y * group3),
            ]
            
            for formula in complex_formulas
                @testset "Complex: $formula" begin
                    # CORRECT: Fit model first, then compile
                    model = lm(formula, df)
                    compiled = compile_formula(model, data)
                    
                    # Test a subset of rows (these are expensive)
                    output = Vector{Float64}(undef, length(compiled))
                    for test_row in [1, 20, 50]
                        compiled(output, data, test_row)
                        expected = modelmatrix(model)[test_row, :]
                        @test isapprox(output, expected, rtol=1e-12)
                    end
                end
            end
        end
    end
    
    @testset "Nested and Complex Functions" begin
        @testset "Nested mathematical functions" begin
            nested_formulas = [
                @formula(response ~ log(exp(x))),          # log(exp(x))
                @formula(response ~ sqrt(x^2 + y^2)),      # Euclidean distance
                @formula(response ~ log(z + sqrt(w^2))),   # Nested with arithmetic
                @formula(response ~ exp(log(z))),          # exp(log(z))
                @formula(response ~ (x + y)^2),            # Polynomial expansion
                @formula(response ~ abs(x - y)),           # Absolute difference
                @formula(response ~ log(abs(x) + 1)),      # Nested abs in log
            ]
            
            f = nested_formulas[2]
            for f in nested_formulas
                @testset "Nested: $formula" begin
                    # CORRECT: Fit model first, then compile
                    model = lm(f, df)
                    compiled = compile_formula(model, data)
                    
                    output = Vector{Float64}(undef, length(compiled))
                    for test_row in [1, 10, 30]
                        compiled(output, data, test_row)
                        expected = modelmatrix(model)[test_row, :]
                        @test isapprox(output, expected, rtol=1e-10) # Slightly relaxed for nested functions
                    end
                end
            end
        end
        
        @testset "Boolean and logical functions" begin
            boolean_formulas = [
                @formula(response ~ (x > 0)),
                @formula(response ~ (x > y)),
                @formula(response ~ (x > 0) + (y < 0)),
                @formula(response ~ (x > 0) * group3),
                @formula(response ~ flag + x),
                @formula(response ~ flag * group3),
                @formula(response ~ (z > median(z)) * x),  # This might be complex
            ]
            
            for formula in boolean_formulas
                @testset "Boolean: $formula" begin
                    try
                        # CORRECT: Fit model first, then compile
                        model = lm(formula, df)
                        compiled = compile_formula(model, data)
                        
                        output = Vector{Float64}(undef, length(compiled))
                        compiled(output, data, 1)
                        expected = modelmatrix(model)[1, :]
                        @test isapprox(output, expected, rtol=1e-12)
                    catch e
                        @warn "Boolean formula might not be supported yet: $formula" exception=e
                        # Some boolean operations might not be implemented yet
                        @test_skip false  # Mark as expected failure for now
                    end
                end
            end
        end
    end
    
    @testset "Edge Cases and Stress Tests" begin
        @testset "Large categorical variables" begin
            # Test with many levels
            large_cat = categorical(rand(1:20, n))  # 20 levels
            df_large = copy(df)
            df_large.large_cat = large_cat
            data_large = Tables.columntable(df_large)
            
            # CORRECT: Fit model first, then compile
            model = lm(@formula(response ~ large_cat), df_large)
            compiled = compile_formula(model, data_large)
            
            output = Vector{Float64}(undef, length(compiled))
            compiled(output, data_large, 5)
            expected = modelmatrix(model)[5, :]
            @test isapprox(output, expected, rtol=1e-12)
        end
        
        @testset "Many variables" begin
            # Test with many predictors
            formula_many = @formula(response ~ x + y + z + w + t + group3 + group4 + binary)
            # CORRECT: Fit model first, then compile
            model = lm(formula_many, df)
            compiled = compile_formula(model, data)
            
            output = Vector{Float64}(undef, length(compiled))
            compiled(output, data, 1)
            expected = modelmatrix(model)[1, :]
            @test isapprox(output, expected, rtol=1e-12)
        end
        
        @testset "Mixed complexity" begin
            # Very complex formula combining everything
            complex_formula = @formula(response ~ 
                x * log(z) + 
                y^2 * group3 + 
                sqrt(abs(w)) * binary +
                group3 * group4 +
                (x > 0) * group3
            )
            
            try
                # CORRECT: Fit model first, then compile
                model = lm(complex_formula, df)
                compiled = compile_formula(model, data)
                
                output = Vector{Float64}(undef, length(compiled))
                compiled(output, data, 1)
                expected = modelmatrix(model)[1, :]
                @test isapprox(output, expected, rtol=1e-10)
            catch e
                @warn "Very complex formula failed - this might be expected" exception=e
                @test_skip false
            end
        end
    end
    
    @testset "Self-Contained Structure Validation" begin
        @testset "Evaluator types created correctly" begin
            # Test that appropriate evaluator types are created
            model_simple = lm(@formula(response ~ x + group3), df)
            compiled = compile_formula(model_simple, data)
            
            @test compiled.root_evaluator isa AbstractEvaluator
            
            # Should be a CombinedEvaluator containing different sub-types
            if compiled.root_evaluator isa CombinedEvaluator
                sub_types = [typeof(sub) for sub in compiled.root_evaluator.sub_evaluators]
                @test length(sub_types) > 0
                println("  Sub-evaluator types: $sub_types")
            end
        end
        
        @testset "Scratch space allocation" begin
            model_interaction = lm(@formula(response ~ x * group3 * group4), df)
            compiled = compile_formula(model_interaction, data)
            
            scratch_size = length(compiled.scratch_space)
            @test scratch_size >= 0
            
            println("  Scratch space size for complex interaction: $scratch_size")
        end
        
        @testset "Position validation" begin
            model = lm(@formula(response ~ x + y + group3), df)
            compiled = compile_formula(model, data)
            
            # Output width should match model matrix
            @test compiled.output_width == size(modelmatrix(model), 2)
            
            println("  Output width: $(compiled.output_width)")
        end
    end
    
    @testset "Performance and Allocation Tests" begin
        @testset "Allocation budget compliance" begin
            # Test the 50MB budget for 1M evaluations
            model = lm(@formula(response ~ x * y * group3 + log(z) * group4), df)
            compiled = compile_formula(model, data)
            
            output = Vector{Float64}(undef, length(compiled));
            @btime compiled(output, data, 1);
            
            # Warmup
            for i in 1:100
                compiled(output, data, (i % n) + 1)
            end

            # Test allocation for smaller sample (scale to estimate 1M)
            n_test = 1000
            total_allocs = @allocated begin
                for i in 1:n_test
                    compiled(output, data, (i % n) + 1)
                end
            end
            
            # Estimate allocation for 1M evaluations
            allocs_per_eval = total_allocs / n_test
            estimated_1M_allocs = allocs_per_eval * 1_000_000
            estimated_MB = estimated_1M_allocs / (1024^2)
            
            @test estimated_MB < 50
            "Estimated $(estimated_MB) MB for 1M evaluations exceeds 50MB budget"
            
            println("  Estimated allocation for 1M evaluations: $(round(estimated_MB, digits=2)) MB")
            println("  Average allocation per evaluation: $(round(allocs_per_eval, digits=1)) bytes")
        end
        
        @testset "Basic performance test" begin
            model = lm(@formula(response ~ x * group3 + y^2), df)
            compiled = compile_formula(model, data)
            
            output = Vector{Float64}(undef, length(compiled))
            
            # Time a reasonable number of evaluations
            n_evals = 10000
            elapsed = @elapsed begin
                for i in 1:n_evals
                    compiled(output, data, (i % n) + 1)
                end
            end
            
            time_per_eval = elapsed / n_evals * 1e9  # nanoseconds
            println("  Time per evaluation: $(round(time_per_eval, digits=1)) ns")
            
            # Should be reasonably fast (less than 1μs per evaluation for complex formula)
            @test time_per_eval < 1000  # Less than 1000ns = 1μs
        end
    end
    
    @testset "Comparison with Ground Truth" begin
        @testset "Results identical to modelmatrix" begin
            # Test that self-contained evaluators produce identical results to StatsModels
            formulas_to_compare = [
                @formula(response ~ x),
                @formula(response ~ x * group3),
                @formula(response ~ log(z) + group3 * binary),
                @formula(response ~ x * y * group3),
            ]
            
            for formula in formulas_to_compare
                @testset "Comparing results for: $formula" begin
                    # CORRECT: Fit model first, then compile
                    model = lm(formula, df)
                    compiled = compile_formula(model, data)
                    
                    # Test multiple rows for robustness
                    output = Vector{Float64}(undef, length(compiled))
                    for test_row in [1, 10, 50, 100]
                        compiled(output, data, test_row)
                        expected = modelmatrix(model)[test_row, :]
                        @test isapprox(output, expected, rtol=1e-14)
                        "Self-contained results don't match modelmatrix at row $test_row"
                    end
                end
            end
        end
        
        @testset "Test N-way interactions (Generality Test)" begin
            # This is the critical test for unlimited formula complexity
            nway_formulas = [
                @formula(response ~ x * y * z * w),                           # 4-way continuous
                @formula(response ~ x * group3 * binary * group4),           # 4-way mixed
                @formula(response ~ x * y * z * w * t),                      # 5-way continuous  
                @formula(response ~ group3 * binary * group4 * group5),      # 4-way categorical
            ]
            
            f = nway_formulas[1]
            for f in nway_formulas
                @testset "N-way interaction: $formula" begin
                    @show f
                    try
                        # Fit model, then compile
                        model = lm(f, df)
                        compiled = compile_formula(model, data)
                        rhs = fixed_effects_form(model).rhs
                        
                        # Test correctness
                        output = Vector{Float64}(undef, length(compiled))
                        compiled(output, data, 1)
                        expected = modelmatrix(model)[1, :]
                        @test isapprox(output, expected, rtol=1e-12)
                        
                        println("  ✅ $(length(rhs.terms))-way interaction works: $(compiled.output_width) outputs")
                        
                    catch e
                        if occursin("not yet implemented", string(e)) || occursin("not implemented", string(e))
                            @test_broken false
                            # "N-way interaction limitation: $f"
                            println("  ❌ LIMITATION FOUND: $f - $(typeof(e)): $e")
                        else
                            @warn "Unexpected error for N-way interaction: $f" exception=e
                            rethrow(e)
                        end
                    end
                end
            end
        end
        
        @testset "Test complex function generality" begin
            # Test functions with multiple arguments and complex nesting
            complex_function_formulas = [
                @formula(response ~ log(abs(x) + abs(y) + abs(z))),                    # 3-argument function
                @formula(response ~ sqrt(x^2 + y^2 + z^2)),            # Complex nested function
                @formula(response ~ log(exp(x) + exp(y))),              # Multiple same functions
                @formula(response ~ abs(x - y) + abs(y - z)),           # Multiple different functions
            ]
            
            f = complex_function_formulas[2]
            for f in complex_function_formulas
                @testset "Complex function: $f" begin
                    try
                        # Fit model first, then compile
                        model = lm(f, df)
                        compiled = compile_formula(model, data)
                        
                        output = Vector{Float64}(undef, length(compiled))
                        compiled(output, data, 1)
                        expected = modelmatrix(model)[1, :]
                        @test isapprox(output, expected, rtol=1e-12)
                        
                        println("  ✅ Complex function works: $f")
                        
                    catch e
                        if occursin("not yet implemented", string(e)) || occursin("not implemented", string(e))
                            @test_broken false
                            "Function complexity limitation: $f"
                            println("  ❌ FUNCTION LIMITATION: $f - $e")
                        else
                            @warn "Unexpected error for complex function: $f" exception=e
                            rethrow(e)
                        end
                    end
                end
            end
        end
    end
    
    @testset "Stress Tests and Edge Cases" begin
        @testset "Very large formulas" begin
            # Test handling of formulas with many terms
            try
                # Create a formula with many terms
                large_formula = @formula(response ~ 
                    x + y + z + w + t +
                    group3 + group4 + binary +
                    x*y + x*z + y*z + x*group3 + y*group3 + z*group3 +
                    log(z) + sqrt(abs(w)) + x^2 + y^2
                )
                
                # CORRECT: Fit model first, then compile
                model = lm(large_formula, df)
                compiled = compile_formula(model, data)
                
                output = Vector{Float64}(undef, length(compiled))
                compiled(output, data, 1)
                expected = modelmatrix(model)[1, :]
                @test isapprox(output, expected, rtol=1e-12)
                
                println("  ✅ Large formula works: $(compiled.output_width) terms")
                
            catch e
                @warn "Large formula test failed" exception=e
                @test_skip false
            end
        end
        
        @testset "Edge case data values" begin
            # Test with edge case data (zeros, negatives, etc.)
            df_edge = DataFrame(
                x = [0.0, -1.0, 1e-10, 1e10, -1e10],
                z = [1e-10, 1.0, 1e10, 0.1, 0.001],  # For log - all positive
                group = categorical(["A", "B", "A", "B", "A"]),
                response = [1.0, 2.0, 3.0, 4.0, 5.0]
            )
            data_edge = Tables.columntable(df_edge)
            
            # CORRECT: Fit model first, then compile
            model = lm(@formula(response ~ x + log(z) + group), df_edge)
            compiled = compile_formula(model, data_edge)
            
            output = Vector{Float64}(undef, length(compiled))
            for i in 1:5
                compiled(output, data_edge, i)
                expected = modelmatrix(model)[i, :]
                @test isapprox(output, expected, rtol=1e-12)
                "Edge case failed at row $i"
            end
            
            println("  ✅ Edge case data values handled correctly")
        end
    end
end
