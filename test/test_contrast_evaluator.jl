# test_contrast_evaluator.jl - Ground truth validation tests for ContrastEvaluator
#
# Testing philosophy: Every test compares ContrastEvaluator results to the ground truth
# of manual data substitution using modelrow!. This validates that CounterfactualVector
# produces identical results to the obviously-correct-but-slow approach of copying data.
#
# Run with:
# julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_contrast_evaluator.jl\")"

using Test, BenchmarkTools
using FormulaCompiler
using DataFrames, GLM, StatsModels, CategoricalArrays, Tables
using Random, Statistics

Random.seed!(06515)

@testset "ContrastEvaluator Ground Truth Validation" begin

    # ================================================================================
    # GROUND TRUTH HELPER FUNCTION
    # ================================================================================

    """
        ground_truth_contrast(compiled, data, row, var, from, to)

    REFERENCE IMPLEMENTATION - Ground truth for all contrast tests.

    This is the obviously correct but slow way to compute contrasts.
    It manually copies data and uses modelrow! - no CounterfactualVector tricks.
    Every ContrastEvaluator result MUST match this function's output.

    CRITICAL: Uses copy() to preserve CategoricalArray structure (see CLAUDE.md).
    This ensures categorical levels and contrast matrices remain intact.
    """
    function ground_truth_contrast(compiled, data, row, var, from, to)
        # Step 1: Evaluate with "from" value
        col_orig = getproperty(data, var)
        col_from = copy(col_orig)  # CRITICAL: Preserve CategoricalArray/type structure
        col_from[row] = from
        data_from = merge(data, NamedTuple{(var,)}((col_from,)))

        pred_from = Vector{Float64}(undef, length(compiled))
        modelrow!(pred_from, compiled, data_from, row)

        # Step 2: Evaluate with "to" value
        col_to = copy(col_orig)  # CRITICAL: Preserve CategoricalArray/type structure
        col_to[row] = to
        data_to = merge(data, NamedTuple{(var,)}((col_to,)))

        pred_to = Vector{Float64}(undef, length(compiled))
        modelrow!(pred_to, compiled, data_to, row)

        # Return contrast: difference between "to" and "from" predictions
        return pred_to .- pred_from
    end

    # ================================================================================
    # TEST DATA GENERATION
    # ================================================================================

    """
        create_test_data(n=100)

    Generate test data with diverse variable types for comprehensive testing.

    Includes:
    - Categorical variables (string, integer, boolean levels)
    - Numeric variables (Float64, Int)
    - Binary variables (0/1, true/false)
    - Complex interactions and transformations
    """
    function create_test_data(n=100)
        df = DataFrame(
            # Continuous numeric
            x = randn(n),
            y = randn(n),
            z = abs.(randn(n)) .+ 0.1,  # Positive for log/sqrt

            # Integer numeric (will be converted to Float64 for derivatives)
            age = rand(18:80, n),

            # Categorical variables with string levels
            treatment = categorical(rand(["Control", "Drug_A", "Drug_B"], n)),
            education = categorical(rand(["HS", "College", "Graduate"], n)),
            region = categorical(rand(["North", "South", "East", "West"], n)),

            # Categorical with integer levels
            age_group = categorical(rand(1:3, n)),

            # Binary numeric (0/1)
            female = rand([0, 1], n),
            employed = rand([0.0, 1.0], n),

            # Boolean variables
            married = rand([true, false], n),

            # Boolean categorical
            urban = categorical(rand([true, false], n)),

            # Response
            outcome = randn(n)
        )
        return df, Tables.columntable(df)
    end

    # ================================================================================
    # STEP 2: CORE CORRECTNESS TESTS - CATEGORICAL VARIABLES
    # ================================================================================

    @testset "1. Categorical Variables - Ground Truth" begin
        df, data = create_test_data(200)

        @testset "String levels (String categorical)" begin
            model = lm(@formula(outcome ~ x + treatment), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment])

            # Test multiple rows to ensure consistency
            for row in [1, 25, 50, 100, 150]
                # ContrastEvaluator result
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :treatment, "Control", "Drug_A")

                # Ground truth result
                gt_result = ground_truth_contrast(compiled, data, row, :treatment, "Control", "Drug_A")

                # CRITICAL TEST: Must match exactly
                @test ce_result ≈ gt_result atol=1e-14
            end
        end

        @testset "Integer levels (Integer categorical)" begin
            model = lm(@formula(outcome ~ x + age_group), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:age_group])

            for row in [1, 25, 50, 100]
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :age_group, 1, 2)

                gt_result = ground_truth_contrast(compiled, data, row, :age_group, 1, 2)

                @test ce_result ≈ gt_result atol=1e-14
            end
        end

        @testset "Boolean levels (Boolean categorical)" begin
            model = lm(@formula(outcome ~ x + urban), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:urban])

            for row in [1, 10, 20, 50]
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :urban, true, false)

                gt_result = ground_truth_contrast(compiled, data, row, :urban, true, false)

                @test ce_result ≈ gt_result atol=1e-14
            end
        end

        @testset "All categorical level pairs" begin
            model = lm(@formula(outcome ~ x + treatment), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment])

            # Comprehensive: Test ALL pairs of levels
            levels = ["Control", "Drug_A", "Drug_B"]
            for from in levels, to in levels
                from == to && continue  # Skip same-level (tested separately)

                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, 1, :treatment, from, to)

                gt_result = ground_truth_contrast(compiled, data, 1, :treatment, from, to)

                @test ce_result ≈ gt_result atol=1e-14
            end
        end
    end

    # ================================================================================
    # STEP 3: NUMERIC AND BINARY VARIABLES
    # ================================================================================

    @testset "2. Numeric Variables - Ground Truth" begin
        df, data = create_test_data(200)

        @testset "Float64 numeric" begin
            model = lm(@formula(outcome ~ x + y), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:x])

            for row in [1, 25, 50, 100]
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :x, 0.0, 1.0)

                gt_result = ground_truth_contrast(compiled, data, row, :x, 0.0, 1.0)

                @test ce_result ≈ gt_result atol=1e-14
            end
        end

        @testset "Integer numeric (converted to Float64)" begin
            model = lm(@formula(outcome ~ x + age), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:age])

            for row in [1, 25, 50, 100]
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :age, 30, 50)

                gt_result = ground_truth_contrast(compiled, data, row, :age, 30, 50)

                @test ce_result ≈ gt_result atol=1e-14
            end
        end
    end

    @testset "3. Binary Variables - Ground Truth" begin
        df, data = create_test_data(200)

        @testset "Binary numeric (0/1)" begin
            model = lm(@formula(outcome ~ x + female), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:female])

            for row in [1, 25, 50, 100]
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :female, 0, 1)

                gt_result = ground_truth_contrast(compiled, data, row, :female, 0, 1)

                @test ce_result ≈ gt_result atol=1e-14
            end
        end

        @testset "Binary boolean (true/false)" begin
            model = lm(@formula(outcome ~ x + married), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:married])

            for row in [1, 25, 50, 100]
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :married, false, true)

                gt_result = ground_truth_contrast(compiled, data, row, :married, false, true)

                @test ce_result ≈ gt_result atol=1e-14
            end
        end
    end

    # ================================================================================
    # STEP 4: INTERACTIONS AND COMPLEX FORMULAS
    # ================================================================================

    @testset "4. Interactions - Ground Truth" begin
        df, data = create_test_data(200)

        @testset "Categorical × Continuous interaction" begin
            # Model: y ~ x * treatment (x * treatment interaction)
            model = lm(@formula(outcome ~ x * treatment), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment])

            for row in [1, 25, 50, 100]
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :treatment, "Control", "Drug_A")

                gt_result = ground_truth_contrast(compiled, data, row, :treatment, "Control", "Drug_A")

                # With interaction, multiple coefficients should differ
                @test ce_result ≈ gt_result atol=1e-14
                @test count(x -> abs(x) > 1e-10, ce_result) >= 2  # Main + interaction
            end
        end

        @testset "Categorical × Categorical interaction" begin
            # Model: y ~ treatment * education (cat-cat interaction)
            model = lm(@formula(outcome ~ treatment * education), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :education])

            for row in [1, 25, 50]
                # Test treatment contrast
                ce_result = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result, evaluator, row, :treatment, "Control", "Drug_A")

                gt_result = ground_truth_contrast(compiled, data, row, :treatment, "Control", "Drug_A")

                @test ce_result ≈ gt_result atol=1e-14

                # Test education contrast
                ce_result_edu = Vector{Float64}(undef, length(compiled))
                contrast_modelrow!(ce_result_edu, evaluator, row, :education, "HS", "Graduate")

                gt_result_edu = ground_truth_contrast(compiled, data, row, :education, "HS", "Graduate")

                @test ce_result_edu ≈ gt_result_edu atol=1e-14
            end
        end
    end

    @testset "5. Complex Formulas - Ground Truth" begin
        df, data = create_test_data(200)

        @testset "Transformations with interactions" begin
            # Model: y ~ log(z) * treatment + sqrt(z) * education
            model = lm(@formula(outcome ~ log(z) * treatment + sqrt(z) * education), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :education])

            for row in [1, 25, 50]
                for (var, from, to) in [
                    (:treatment, "Control", "Drug_A"),
                    (:education, "HS", "College")
                ]
                    ce_result = Vector{Float64}(undef, length(compiled))
                    contrast_modelrow!(ce_result, evaluator, row, var, from, to)

                    gt_result = ground_truth_contrast(compiled, data, row, var, from, to)

                    @test ce_result ≈ gt_result atol=1e-14
                end
            end
        end
    end

    # ================================================================================
    # STEP 5: EDGE CASES
    # ================================================================================

    @testset "6. Edge Cases - Ground Truth" begin
        df, data = create_test_data(200)

        @testset "Same level contrast (should be zeros)" begin
            model = lm(@formula(outcome ~ x + treatment), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment])

            ce_result = Vector{Float64}(undef, length(compiled))
            contrast_modelrow!(ce_result, evaluator, 1, :treatment, "Control", "Control")

            gt_result = ground_truth_contrast(compiled, data, 1, :treatment, "Control", "Control")

            @test ce_result ≈ gt_result atol=1e-14
            @test all(x -> abs(x) < 1e-14, ce_result)  # Should be exactly zero
        end

        @testset "Alternative contrast coding (EffectsCoding)" begin
            # Test one alternative coding to validate pattern works
            # DummyCoding tested comprehensively elsewhere
            df_effects = copy(df)
            data_effects = Tables.columntable(df_effects)

            # Use EffectsCoding instead of default DummyCoding
            model_effects = lm(@formula(outcome ~ x + treatment), df_effects,
                              contrasts = Dict(:treatment => EffectsCoding()))
            compiled_effects = compile_formula(model_effects, data_effects)
            evaluator_effects = contrastevaluator(compiled_effects, data_effects, [:treatment])

            # Validate against ground truth
            ce_result = Vector{Float64}(undef, length(compiled_effects))
            contrast_modelrow!(ce_result, evaluator_effects, 1, :treatment, "Control", "Drug_A")

            gt_result = ground_truth_contrast(compiled_effects, data_effects, 1, :treatment, "Control", "Drug_A")

            @test ce_result ≈ gt_result atol=1e-14
        end
    end

    # ================================================================================
    # STEP 6: PERFORMANCE VALIDATION (AFTER CORRECTNESS)
    # ================================================================================

    @testset "7. Performance - Zero Allocations" begin
        df, data = create_test_data(200)

        # Note: CategoricalArray getindex inherently allocates ~32 bytes per CategoricalValue.
        # This is baseline CategoricalArrays.jl behavior, not FormulaCompiler overhead.
        # To achieve true zero allocations, test each variable type with dedicated evaluators.

        @testset "Categorical zero allocations (single variable)" begin
            model = lm(@formula(outcome ~ x + treatment), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment])
            buf = Vector{Float64}(undef, length(compiled))

            # Warm up
            contrast_modelrow!(buf, evaluator, 1, :treatment, "Control", "Drug_A")

            # Test zero allocations with single categorical variable
            result = @benchmark contrast_modelrow!($buf, $evaluator, 1, :treatment, "Control", "Drug_A")
            @test result.memory == 0
            @test result.allocs == 0
        end

        @testset "Numeric zero allocations (single variable)" begin
            model = lm(@formula(outcome ~ x + female), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:x])
            buf = Vector{Float64}(undef, length(compiled))

            # Warm up
            contrast_modelrow!(buf, evaluator, 1, :x, 0.0, 1.0)

            result = @benchmark contrast_modelrow!($buf, $evaluator, 1, :x, 0.0, 1.0)
            @test result.memory == 0
            @test result.allocs == 0
        end

        @testset "Binary zero allocations (single variable)" begin
            model = lm(@formula(outcome ~ x + female), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:female])
            buf = Vector{Float64}(undef, length(compiled))

            # Warm up
            contrast_modelrow!(buf, evaluator, 1, :female, 0, 1)

            result = @benchmark contrast_modelrow!($buf, $evaluator, 1, :female, 0, 1)
            @test result.memory == 0
            @test result.allocs == 0
        end

        @testset "Multiple numeric variables - zero allocations" begin
            # Pure numeric (no categoricals) can achieve zero allocations even with multiple vars
            df_numeric = DataFrame(
                outcome = randn(200),
                x1 = randn(200),
                x2 = randn(200),
                x3 = randn(200)
            )
            data_numeric = Tables.columntable(df_numeric)
            model = lm(@formula(outcome ~ x1 + x2 + x3), df_numeric)
            compiled = compile_formula(model, data_numeric)
            evaluator = contrastevaluator(compiled, data_numeric, [:x1, :x2, :x3])
            buf = Vector{Float64}(undef, length(compiled))

            contrast_modelrow!(buf, evaluator, 1, :x1, 0.0, 1.0)  # warmup
            result = @benchmark contrast_modelrow!($buf, $evaluator, 1, :x1, 0.0, 1.0)
            @test result.memory == 0
            @test result.allocs == 0
        end

        @testset "Multi-variable with categoricals - performance characteristics" begin
            # When evaluator includes categorical variables, performance characteristics may vary
            # depending on Julia's ability to optimize CategoricalValue operations
            model = lm(@formula(outcome ~ x + treatment + female), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :female, :x])
            buf = Vector{Float64}(undef, length(compiled))

            # Categorical contrasts may allocate depending on compiler optimizations
            contrast_modelrow!(buf, evaluator, 1, :treatment, "Control", "Drug_A")
            result_cat = @benchmark contrast_modelrow!($buf, $evaluator, 1, :treatment, "Control", "Drug_A")

            # Numeric variable contrasts
            contrast_modelrow!(buf, evaluator, 1, :x, 0.0, 1.0)
            result_num = @benchmark contrast_modelrow!($buf, $evaluator, 1, :x, 0.0, 1.0)

            # The key result: both should be relatively low allocation
            # (either 0 or small CategoricalValue-related allocations)
            @test result_cat.memory <= 128  # May be 0 or ~32-64 bytes for categorical operations
            @test result_num.memory <= 128  # Should match categorical performance

            # Performance should still be good (sub-microsecond)
            @test minimum(result_cat.times) < 1_000_000  # < 1ms
            @test minimum(result_num.times) < 1_000_000  # < 1ms
        end
    end

    # ================================================================================
    # STEP 7: ERROR HANDLING
    # ================================================================================

    @testset "8. Error Handling" begin
        df, data = create_test_data(200)
        model = lm(@formula(outcome ~ x + treatment), df)
        compiled = compile_formula(model, data)
        evaluator = contrastevaluator(compiled, data, [:treatment])
        buf = Vector{Float64}(undef, length(compiled))

        @testset "Invalid variable name" begin
            @test_throws ErrorException contrast_modelrow!(buf, evaluator, 1, :nonexistent, "A", "B")
        end

        @testset "Invalid categorical level" begin
            @test_throws ErrorException contrast_modelrow!(buf, evaluator, 1, :treatment, "Invalid", "Drug_A")
            @test_throws ErrorException contrast_modelrow!(buf, evaluator, 1, :treatment, "Control", "Invalid")
        end

        @testset "Buffer size mismatch" begin
            wrong_buf = Vector{Float64}(undef, 5)
            @test_throws DimensionMismatch contrast_modelrow!(wrong_buf, evaluator, 1, :treatment, "Control", "Drug_A")
        end
    end

    # ================================================================================
    # STEP 8: MATHEMATICAL PROPERTIES (SECONDARY VALIDATION)
    # ================================================================================

    @testset "9. Mathematical Properties (Secondary Validation)" begin
        df, data = create_test_data(200)
        model = lm(@formula(outcome ~ x + treatment + education), df)
        compiled = compile_formula(model, data)
        evaluator = contrastevaluator(compiled, data, [:treatment, :education])

        # AFTER ground truth validation, these provide additional confidence

        @testset "Symmetry property" begin
            # contrast(A,B) = -contrast(B,A)
            for row in [1, 25, 50]
                contrast_AB = Vector{Float64}(undef, length(compiled))
                contrast_BA = Vector{Float64}(undef, length(compiled))

                contrast_modelrow!(contrast_AB, evaluator, row, :treatment, "Control", "Drug_A")
                contrast_modelrow!(contrast_BA, evaluator, row, :treatment, "Drug_A", "Control")

                @test contrast_AB ≈ -contrast_BA atol=1e-14
            end
        end

        @testset "Transitivity property" begin
            # contrast(A,C) = contrast(A,B) + contrast(B,C)
            levels = ["HS", "College", "Graduate"]

            contrast_AC = Vector{Float64}(undef, length(compiled))
            contrast_AB = Vector{Float64}(undef, length(compiled))
            contrast_BC = Vector{Float64}(undef, length(compiled))

            contrast_modelrow!(contrast_AC, evaluator, 1, :education, "HS", "Graduate")
            contrast_modelrow!(contrast_AB, evaluator, 1, :education, "HS", "College")
            contrast_modelrow!(contrast_BC, evaluator, 1, :education, "College", "Graduate")

            @test contrast_AC ≈ contrast_AB .+ contrast_BC atol=1e-12
        end
    end

end  # @testset "ContrastEvaluator Ground Truth Validation"

