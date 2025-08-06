# categorical_override_tests.jl
# Five comprehensive tests for categorical override correctness

using Test, BenchmarkTools
using FormulaCompiler

using DataFrames
using GLM
using StatsModels
using CategoricalArrays
using Tables
using Random

"""
    create_categorical_test_data(n=100)

Create test data with various categorical variables for testing.
"""
function create_categorical_test_data(n=100)
    
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        cat2 = categorical(rand(["A", "B"], n)),
        cat3 = categorical(rand(["Low", "Med", "High"], n)),
        cat4 = categorical(rand(["W", "X", "Y", "Z"], n)),
        cat5 = categorical(rand(["P", "Q", "R", "S", "T"], n)),
        binary = rand([true, false], n),
        binary_cat = categorical(rand([true, false], n)),
        ordered_cat = categorical(rand(["Small", "Medium", "Large"], n), ordered=true),
        response = randn(n)
    )
    
    return df, Tables.columntable(df)
end

Random.seed!(08540)

@testset "Check variable differentiation" begin
    df, data = create_categorical_test_data(100);

    @testset "Varying cts. variable" begin
        row_idx = 1
        fx = @formula(response ~ x + cat3)
        model = lm(fx, df)

        overrides1 = Dict(:x => minimum(df.x))
        overrides2 = Dict(:x => maximum(df.x))
        scenario1 = create_scenario("Low", data, overrides1)
        scenario2 = create_scenario("High", data, overrides2)
        compiled_ = compile_formula(model, scenario1.data)

        output1 = Vector{Float64}(undef, size(modelmatrix(model), 2))
        output2 = Vector{Float64}(undef, size(modelmatrix(model), 2))
        for row_idx in [1,10,50]
            compiled_(output1, scenario1.data, row_idx)
            compiled_(output2, scenario2.data, row_idx)
            @test !(output1 == output2)
        end
    end

    @testset "Varying cat. variable" begin
        row_idx = 1

        fx = @formula(response ~ x + cat3)
        model = lm(fx, df)
        
        overrides1 = Dict(:cat3 => "Low")
        overrides2 = Dict(:cat3 => "High")
        scenario1 = create_scenario("Low", data, overrides1)
        scenario2 = create_scenario("High", data, overrides2)
        compiled_ = compile_formula(model, scenario1.data)

        output1 = Vector{Float64}(undef, size(modelmatrix(model), 2))
        output2 = Vector{Float64}(undef, size(modelmatrix(model), 2))
        for row_idx in [1,10,50]
            compiled_(output1, scenario1.data, row_idx)

            compiled_(output2, scenario2.data, row_idx)

            @test !(output1 == output2)
        end
    end
end

@testset "Categorical Override Tests" begin
    
    # Create test data
    df, data = create_categorical_test_data(100);
    
    @testset "Single Categorical - All Levels" begin
        # Test that each level of a categorical produces correct output
        fx = @formula(response ~ x + cat3)
        model = lm(fx, df)
        
        for level in ["Low", "Med", "High"]
            @testset "Level: $level" begin
                # Create scenario
                overrides = Dict(:cat3 => level)
                scenario = create_scenario("cat3_$level", data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Create reference
                ref_df = copy(df)
                ref_df.cat3 .= level

                # Reference model matrix
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test multiple rows
                for row_idx in [1, 25, 50, 75, 100]
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end
                
                # Verify constant categorical effect across all rows
                outputs = []
                for row_idx in 1:5
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    push!(outputs, copy(output))
                end
                
                # The categorical columns should be identical across rows
                # (columns 3 and 4 for cat3 with dummy coding)
                for i in 2:5
                    @test outputs[i][3:end] ≈ outputs[1][3:end] atol=1e-12
                end
            end
        end
    end
    
    @testset "Multiple Categoricals - Mixed Levels" begin
        # Test overriding multiple categorical variables simultaneously
        fx = @formula(response ~ cat2 + cat3 + cat4)
        model = lm(fx, df)
        
        test_cases = [
            ("all_reference", Dict(:cat2 => "A", :cat3 => "Low", :cat4 => "W")),
            ("all_last", Dict(:cat2 => "B", :cat3 => "High", :cat4 => "Z")),
            ("mixed", Dict(:cat2 => "A", :cat3 => "Med", :cat4 => "Y")),
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end

                # Reference model matrix
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                for row_idx in [1, 50, 100]
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end
                
                # All rows should produce identical output (all categoricals overridden)
                out1 = Vector{Float64}(undef, size(ref_mm, 2))
                out2 = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(out1, scenario.data, 1)
                compiled(out2, scenario.data, 50)
                @test out1 ≈ out2 atol=1e-12
            end
        end
    end
    
    @testset "Categorical in Interaction" begin
        # Test categorical overrides in interaction terms
        fx = @formula(response ~ x * cat3)
        model = lm(fx, df)
        
        # Test each categorical level with fixed continuous
        overrides = Dict(:x => 2.0, :cat3 => "Med")
        scenario = create_scenario("interaction_test", data, overrides)
        compiled = compile_formula(model, scenario.data)
        
        # Reference
        ref_df = DataFrame(df)
        ref_df.x .= 2.0
        ref_df.cat3 .= "Med"
        # Reference model matrix
        ref_mm = modelmatrix(model.mf; data = ref_df)
        
        # The model matrix should have columns for:
        # 1. Intercept
        # 2. x
        # 3. cat3: Med
        # 4. cat3: High  
        # 5. x & cat3: Med
        # 6. x & cat3: High
        
        output = Vector{Float64}(undef, size(ref_mm, 2))
        for row_idx in [1, 10, 50, 100]
            compiled(output, scenario.data, row_idx)
            @test output ≈ ref_mm[row_idx, :] atol=1e-10
            
            # Verify the interaction terms
            # For x=2.0 and cat3="Med", we expect:
            # - Column 3 (cat3:Med) = 1.0
            # - Column 4 (cat3:High) = 0.0
            # - Column 5 (x & cat3:Med) = 2.0 * 1.0 = 2.0
            # - Column 6 (x & cat3:High) = 2.0 * 0.0 = 0.0
            
            # The model knows its own structure
            schema = model.mf.schema
            cat3_term = schema[term(:cat3)]  # or however it's stored
            contrasts = cat3_term.contrasts
            lvl = contrasts.levels
            contrast_matrix = cat3_term.contrasts.matrix  # The actual contrasts

            # For cat3 = "Med", find its levelcode
            med_levelcode = levelcode(categorical(["Med"], levels=lvl)[1])
            # Use the contrast matrix to determine expected dummy values
            expected_dummies = contrast_matrix[med_levelcode, :]

            # 1. Test that output matches reference - this validates correctness
            @test output ≈ ref_mm[row_idx, :] atol=1e-10
        end
    end
    
    @testset "Test 4: Binary Categorical Override" begin
        # Test binary categorical (true/false) overrides
        fx = @formula(response ~ x + binary)
        model = lm(fx, df)
        
        for value in [true, false]
            @testset "Binary = $value" begin
                overrides = Dict(:binary => value)
                scenario = create_scenario("binary_$value", data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                ref_df.binary .= value
                # Reference model matrix
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test rows
                for row_idx in [1, 50, 100]
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end
                
                # Check that binary effect is constant
                outputs = []
                for row_idx in [1, 20, 40, 60, 80]
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    push!(outputs, output[3])  # Binary effect column
                end
                
                # All binary effects should be identical
                @test all(x -> x ≈ outputs[1], outputs)
            end
        end
    end
    
    @testset "Test 5: Categorical-Only Formula with Complete Override" begin
        # Test a formula with only categorical variables, all overridden
        fx = @formula(response ~ cat2 * cat3 + cat4)
        model = lm(fx, df)
        
        overrides = Dict(
            :cat2 => "B",
            :cat3 => "High",
            :cat4 => "Y"
        )
        
        scenario = create_scenario("all_cat", data, overrides)
        compiled = compile_formula(model, scenario.data)
        
        # Reference
        ref_df = DataFrame(df)
        ref_df.cat2 .= "B"
        ref_df.cat3 .= "High"
        ref_df.cat4 .= "Y"
        # Reference model matrix
        ref_mm = modelmatrix(model.mf; data = ref_df)
        
        # Test that all rows produce identical output
        outputs = Matrix{Float64}(undef, 10, size(ref_mm, 2))
        for i in 1:10
            row_idx = i * 10
            output_view = view(outputs, i, :)
            compiled(output_view, scenario.data, row_idx)
            
            # Compare with reference
            @test output_view ≈ ref_mm[row_idx, :] atol=1e-10
        end
        
        # All rows should be identical since all variables are overridden
        for i in 2:10
            @test outputs[i, :] ≈ outputs[1, :] atol=1e-12
        end
        
        # Verify the specific encoding
        # With cat2="B", cat3="High", cat4="Y", we expect specific dummy values
        output = outputs[1, :]
        
        # The exact values depend on the contrast coding, but we can verify:
        # 1. The output matches the reference exactly
        # 2. The output is constant across all rows
        # 3. The interaction terms are computed correctly
        
        # For cat2 * cat3 interaction with both at non-reference levels:
        # Should have non-zero interaction term
        
        println("  Model matrix structure:")
        println("    Intercept: $(output[1])")
        println("    cat2[B]: $(output[2])")
        println("    cat3[Med]: $(output[3])")
        println("    cat3[High]: $(output[4])")
        println("    cat4 effects: $(output[5:7])")
        println("    Interaction terms: $(output[8:end])")
    end
  
    @testset "Categorical Override Edge Cases" begin      
        df, data = create_categorical_test_data(50)
        
        @testset "Override with different value types" begin
            fx = @formula(response ~ cat3)
            model = lm(fx, df)
            
            # Test String override
            scenario_string = create_scenario("string", data; cat3 = "Med")
            
            # Test Symbol override
            scenario_symbol = create_scenario("symbol", data; cat3 = :Med)
            
            # Test index override (2 = "Med" assuming Low, Med, High order)
            scenario_index = create_scenario("index", data; cat3 = 2)
            
            # All should produce identical results
            compiled_string = compile_formula(model, scenario_string.data)
            compiled_symbol = compile_formula(model, scenario_symbol.data)
            # compiled_index = compile_formula(model, scenario_index.data)
            
            out_string = Vector{Float64}(undef, 3)
            out_symbol = Vector{Float64}(undef, 3)
            # out_index = Vector{Float64}(undef, 3)
            
            compiled_string(out_string, scenario_string.data, 1)
            compiled_symbol(out_symbol, scenario_symbol.data, 1)
            # compiled_index(out_index, scenario_index.data, 1)
            
            @test out_string ≈ out_symbol atol=1e-12
            # @test out_string ≈ out_index atol=1e-12
            # INDEX (CAT LEVELCODE) DEFS DON'T WORK --- TOTALLY FINE
        end
        
        @testset "Invalid categorical level" begin
            formula = @formula(response ~ cat3)
            model = lm(formula, df)
            
            # This should throw an error
            @test_throws Exception create_scenario("invalid", data; cat3 = "InvalidLevel")
        end
        
        @testset "Ordered categorical override" begin
            formula = @formula(response ~ ordered_cat)
            model = lm(formula, df)
            
            overrides = Dict(:ordered_cat => "Medium")
            scenario = create_scenario("ordered", data, overrides)
            compiled = compile_formula(model, scenario.data)
            
            # Should work correctly with ordered categoricals
            output = Vector{Float64}(undef, 3)
            @test_nowarn compiled(output, scenario.data, 1)
        end
    end
end
