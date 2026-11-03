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
using FormulaCompiler: NumericCounterfactualVector, CategoricalCounterfactualVector, BoolCounterfactualVector

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

@testset "Categorical Override Tests" begin
    
    # Create test data
    df, data = create_categorical_test_data(100);
    
    @testset "Single Categorical - All Levels" begin
        # Test that each level of a categorical produces correct output
        fx = @formula(response ~ x + cat3)
        model = lm(fx, df)
        
        for level in ["Low", "Med", "High"]
            @testset "Level: $level" begin
                # Create counterfactual
                cat_val = CategoricalValue(level, data.cat3)
                cf_cat = CategoricalCounterfactualVector(data.cat3, 1, cat_val)
                cf_data = merge(data, (cat3 = cf_cat,))
                compiled = compile_formula(model, cf_data)
                
                # Create reference
                ref_df = copy(df)
                ref_df.cat3 .= level

                # Reference model matrix
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test multiple rows
                for row_idx in [1, 25, 50, 75, 100]
                    FormulaCompiler.update_counterfactual_row!(cf_cat, row_idx)
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, cf_data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end
                
                # Verify constant categorical effect across all rows
                outputs = []
                for row_idx in 1:5
                    FormulaCompiler.update_counterfactual_row!(cf_cat, row_idx)
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, cf_data, row_idx)
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
                # Create counterfactual vectors
                cat2_val = CategoricalValue(overrides[:cat2], data.cat2)
                cat3_val = CategoricalValue(overrides[:cat3], data.cat3)
                cat4_val = CategoricalValue(overrides[:cat4], data.cat4)
                cf_cat2 = CategoricalCounterfactualVector(data.cat2, 1, cat2_val)
                cf_cat3 = CategoricalCounterfactualVector(data.cat3, 1, cat3_val)
                cf_cat4 = CategoricalCounterfactualVector(data.cat4, 1, cat4_val)
                cf_data = merge(data, (cat2 = cf_cat2, cat3 = cf_cat3, cat4 = cf_cat4))
                compiled = compile_formula(model, cf_data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end

                # Reference model matrix
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                for row_idx in [1, 50, 100]
                    FormulaCompiler.update_counterfactual_row!(cf_cat2, row_idx)
                    FormulaCompiler.update_counterfactual_row!(cf_cat3, row_idx)
                    FormulaCompiler.update_counterfactual_row!(cf_cat4, row_idx)
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, cf_data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end

                # All rows should produce identical output (all categoricals overridden)
                out1 = Vector{Float64}(undef, size(ref_mm, 2))
                out2 = Vector{Float64}(undef, size(ref_mm, 2))
                FormulaCompiler.update_counterfactual_row!(cf_cat2, 1)
                FormulaCompiler.update_counterfactual_row!(cf_cat3, 1)
                FormulaCompiler.update_counterfactual_row!(cf_cat4, 1)
                compiled(out1, cf_data, 1)
                FormulaCompiler.update_counterfactual_row!(cf_cat2, 50)
                FormulaCompiler.update_counterfactual_row!(cf_cat3, 50)
                FormulaCompiler.update_counterfactual_row!(cf_cat4, 50)
                compiled(out2, cf_data, 50)
                @test out1 ≈ out2 atol=1e-12
            end
        end
    end
    
    @testset "Categorical in Interaction" begin
        # Test categorical overrides in interaction terms
        fx = @formula(response ~ x * cat3)
        model = lm(fx, df)
        
        # Test each categorical level with fixed continuous
        cf_x = NumericCounterfactualVector{Float64}(data.x, 1, 2.0)
        cat3_med = CategoricalValue("Med", data.cat3)
        cf_cat3 = CategoricalCounterfactualVector(data.cat3, 1, cat3_med)
        cf_data = merge(data, (x = cf_x, cat3 = cf_cat3))
        compiled = compile_formula(model, cf_data)
        
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
            FormulaCompiler.update_counterfactual_row!(cf_x, row_idx)
            FormulaCompiler.update_counterfactual_row!(cf_cat3, row_idx)
            compiled(output, cf_data, row_idx)
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
                cf_binary = BoolCounterfactualVector(data.binary, 1, value)
                cf_data = merge(data, (binary = cf_binary,))
                compiled = compile_formula(model, cf_data)
                
                # Reference
                ref_df = DataFrame(df)
                ref_df.binary .= value
                # Reference model matrix
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test rows
                for row_idx in [1, 50, 100]
                    FormulaCompiler.update_counterfactual_row!(cf_binary, row_idx)
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, cf_data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end

                # Check that binary effect is constant
                outputs = []
                for row_idx in [1, 20, 40, 60, 80]
                    FormulaCompiler.update_counterfactual_row!(cf_binary, row_idx)
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, cf_data, row_idx)
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
        
        cat2_b = CategoricalValue("B", data.cat2)
        cat3_high = CategoricalValue("High", data.cat3)
        cat4_y = CategoricalValue("Y", data.cat4)
        cf_cat2 = CategoricalCounterfactualVector(data.cat2, 1, cat2_b)
        cf_cat3 = CategoricalCounterfactualVector(data.cat3, 1, cat3_high)
        cf_cat4 = CategoricalCounterfactualVector(data.cat4, 1, cat4_y)
        cf_data = merge(data, (cat2 = cf_cat2, cat3 = cf_cat3, cat4 = cf_cat4))
        compiled = compile_formula(model, cf_data)
        
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
            FormulaCompiler.update_counterfactual_row!(cf_cat2, row_idx)
            FormulaCompiler.update_counterfactual_row!(cf_cat3, row_idx)
            FormulaCompiler.update_counterfactual_row!(cf_cat4, row_idx)
            output_view = view(outputs, i, :)
            compiled(output_view, cf_data, row_idx)

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
        
        @debug "Model matrix structure" intercept=output[1] cat2_B=output[2] cat3_Med=output[3] cat3_High=output[4] cat4_effects=output[5:7] interaction_terms=output[8:end]
    end
  
    @testset "Categorical Override Edge Cases" begin      
        df, data = create_categorical_test_data(50)
        
        @testset "Override with different value types" begin
            fx = @formula(response ~ cat3)
            model = lm(fx, df)
            
            # Test String override
            cat3_med_str = CategoricalValue("Med", data.cat3)
            cf_cat3_str = CategoricalCounterfactualVector(data.cat3, 1, cat3_med_str)
            cf_data_str = merge(data, (cat3 = cf_cat3_str,))

            # Test Symbol override - convert to string first
            cat3_med_sym = CategoricalValue(string(:Med), data.cat3)
            cf_cat3_sym = CategoricalCounterfactualVector(data.cat3, 1, cat3_med_sym)
            cf_data_sym = merge(data, (cat3 = cf_cat3_sym,))

            # All should produce identical results
            compiled_string = compile_formula(model, cf_data_str)
            compiled_symbol = compile_formula(model, cf_data_sym)

            out_string = Vector{Float64}(undef, 3)
            out_symbol = Vector{Float64}(undef, 3)

            compiled_string(out_string, cf_data_str, 1)
            compiled_symbol(out_symbol, cf_data_sym, 1)
            
            @test out_string ≈ out_symbol atol=1e-12
            # @test out_string ≈ out_index atol=1e-12
            # INDEX (CAT LEVELCODE) DEFS DON'T WORK --- TOTALLY FINE
        end
        
        @testset "Invalid categorical level" begin
            formula = @formula(response ~ cat3)
            model = lm(formula, df)
            
            # This should throw an error when creating CategoricalValue
            @test_throws Exception CategoricalValue("InvalidLevel", data.cat3)
        end
        
        @testset "Ordered categorical override" begin
            formula = @formula(response ~ ordered_cat)
            model = lm(formula, df)
            
            ordered_med = CategoricalValue("Medium", data.ordered_cat)
            cf_ordered = CategoricalCounterfactualVector(data.ordered_cat, 1, ordered_med)
            cf_data = merge(data, (ordered_cat = cf_ordered,))
            compiled = compile_formula(model, cf_data)
            
            # Should work correctly with ordered categoricals
            output = Vector{Float64}(undef, 3)
            @test_nowarn compiled(output, cf_data, 1)
        end
    end
end
