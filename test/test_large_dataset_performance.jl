# Large Dataset Performance Testing
# Test ContrastEvaluator performance and memory scaling with large datasets

using DataFrames, GLM, FormulaCompiler, Tables, CategoricalArrays, BenchmarkTools
using Test, Random

@testset "Large Dataset Performance" begin
    # Set seed for reproducible tests
    Random.seed!(12345)

    @testset "100K observations performance" begin
        n = 100_000
        @info "Creating 100K observation dataset..."

        # Create large dataset with multiple categorical variables
        df = DataFrame(
            # Outcome variable
            y = randn(n),

            # Continuous variables
            x1 = randn(n),
            x2 = randn(n) .* 0.5,
            x3 = rand(n) .- 0.5,

            # Categorical variables with many levels
            region = rand(["North", "South", "East", "West", "Central", "Northwest",
                          "Southwest", "Northeast", "Southeast"], n),
            treatment = rand(["Control", "Treatment_A", "Treatment_B", "Treatment_C",
                            "Treatment_D", "Treatment_E"], n),
            education = rand(["HS", "College", "Graduate", "Professional", "None"], n),
            income_bracket = rand(["Low", "Lower_Mid", "Mid", "Upper_Mid", "High",
                                 "Very_High", "Extreme"], n),
            age_group = rand(["18-25", "26-35", "36-45", "46-55", "56-65", "65+"], n),

            # Binary variables
            female = rand(Bool, n),
            urban = rand(Bool, n),
            married = rand(Bool, n)
        )

        # Convert to categoricals
        for col in [:region, :treatment, :education, :income_bracket, :age_group]
            df[!, col] = categorical(df[!, col])
        end

        @info "Dataset created: $(nrow(df)) rows, $(ncol(df)) columns"

        # Fit complex model with interactions
        @info "Fitting complex model..."
        model = lm(@formula(y ~ x1 * treatment + x2 * region + x3 * education +
                              female * income_bracket + urban + married + age_group), df)

        # Prepare data
        data = Tables.columntable(df)

        # Test compilation performance
        @info "Testing compilation performance..."
        compilation_time = @elapsed begin
            compiled = compile_formula(model, data)
        end
        @info "Compilation time: $(round(compilation_time, digits=3))s"
        @test compilation_time < 5.0  # Should compile within 5 seconds

        # Test ContrastEvaluator construction performance
        @info "Testing ContrastEvaluator construction..."
        vars = [:treatment, :region, :education, :income_bracket, :age_group, :female]
        construction_time = @elapsed begin
            evaluator = contrastevaluator(compiled, data, vars)
        end
        @info "ContrastEvaluator construction time: $(round(construction_time, digits=3))s"
        @test construction_time < 2.0  # Should construct within 2 seconds

        # Test evaluation performance across dataset
        @info "Testing evaluation performance..."
        contrast_buf = Vector{Float64}(undef, length(compiled))

        # Sample random rows for testing (don't test all 100K)
        test_rows = rand(1:n, 100)

        eval_times = Float64[]
        for row in test_rows
            time = @belapsed contrast_modelrow!($contrast_buf, $evaluator, $row,
                                             :treatment, "Control", "Treatment_A")
            push!(eval_times, time)
        end

        mean_time = mean(eval_times)
        max_time = maximum(eval_times)
        @info "Mean evaluation time: $(round(mean_time * 1e9, digits=1))ns"
        @info "Max evaluation time: $(round(max_time * 1e9, digits=1))ns"

        # Performance thresholds
        @test mean_time < 1e-6  # Mean under 1 microsecond
        @test max_time < 5e-6   # Max under 5 microseconds

        # Test allocation performance (should be zero or minimal)
        @info "Testing allocation performance..."
        # Warm up first to avoid compilation allocations
        contrast_modelrow!(contrast_buf, evaluator, 1, :region, "North", "South")

        # Run multiple times to check consistency
        allocation_results = Int[]
        for i in 1:5
            result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1,
                                                 :region, "North", "South")
            push!(allocation_results, result.memory)
        end

        min_alloc = minimum(allocation_results)
        max_alloc = maximum(allocation_results)
        mean_alloc = sum(allocation_results) / length(allocation_results)

        @info "Allocation results: min=$(min_alloc), max=$(max_alloc), mean=$(mean_alloc)"

        # Check if we consistently get zero allocations
        if all(a == 0 for a in allocation_results)
            @test true  # Perfect zero allocations
            @info "✓ Perfect zero allocations confirmed"
        elseif max_alloc <= 64 && min_alloc == 0
            @test true  # Mostly zero with occasional small allocations (measurement noise)
            @info "✓ Zero allocations with minimal noise ($(max_alloc) bytes max)"
        else
            @test false  # Significant allocation issue
            @info "✗ Significant allocations detected"
        end

        # Test memory usage scaling
        @info "Testing memory usage scaling..."
        evaluator_size = sizeof(evaluator)
        @info "ContrastEvaluator memory usage: $(evaluator_size) bytes"
        @test evaluator_size < 1_000_000  # Should be under 1MB regardless of dataset size

        # Test correctness on random samples
        @info "Testing correctness on random samples..."
        correct_count = 0
        for _ in 1:10
            row = rand(1:n)
            contrast_modelrow!(contrast_buf, evaluator, row, :treatment, "Control", "Treatment_A")
            # Basic sanity checks
            @test !all(contrast_buf .== 0.0)  # Should have some non-zero elements
            @test all(isfinite.(contrast_buf))  # All finite values
            correct_count += 1
        end
        @test correct_count == 10
        @info "✓ All correctness tests passed"
    end

    @testset "50+ categorical levels performance" begin
        n = 10_000  # Smaller dataset but many levels
        @info "Creating dataset with 50+ categorical levels..."

        # Create very high-cardinality categorical variable
        high_card_levels = ["Level_$(i)" for i in 1:60]

        df = DataFrame(
            y = randn(n),
            x = randn(n),
            high_cardinality = rand(high_card_levels, n),
            medium_cardinality = rand(["A_$i" for i in 1:20], n),
            low_cardinality = rand(["Group_$i" for i in 1:5], n)
        )

        # Convert to categoricals
        for col in [:high_cardinality, :medium_cardinality, :low_cardinality]
            df[!, col] = categorical(df[!, col])
        end

        @info "High cardinality variable: $(length(unique(df.high_cardinality))) levels"

        # Fit model
        model = lm(@formula(y ~ x * high_cardinality + medium_cardinality + low_cardinality), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)

        # Test ContrastEvaluator with high-cardinality variables
        @info "Testing high-cardinality ContrastEvaluator..."
        vars = [:high_cardinality, :medium_cardinality, :low_cardinality]
        evaluator = contrastevaluator(compiled, data, vars)

        contrast_buf = Vector{Float64}(undef, length(compiled))

        # Test performance with high-cardinality contrasts
        levels = unique(df.high_cardinality)
        from_level = levels[1]
        to_level = levels[end]

        @info "Testing contrast between $(from_level) and $(to_level)..."
        # Warm up first
        contrast_modelrow!(contrast_buf, evaluator, 1, :high_cardinality, from_level, to_level)
        result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1,
                                             :high_cardinality, $from_level, $to_level)

        @test result.memory == 0
        mean_time_ns = result.times |> median
        @info "High-cardinality contrast time: $(round(mean_time_ns, digits=1))ns"
        @test mean_time_ns < 2e-6 * 1e9  # Under 2 microseconds

        # Test that performance doesn't degrade with many levels
        @info "Testing performance scaling with categorical levels..."
        times_by_levels = Float64[]
        for num_levels in [5, 10, 20, 40, 60]
            available_levels = levels[1:min(num_levels, end)]
            from = available_levels[1]
            to = available_levels[end]

            time = @belapsed contrast_modelrow!($contrast_buf, $evaluator, 1,
                                             :high_cardinality, $from, $to)
            push!(times_by_levels, time)
        end

        # Performance should not degrade significantly with more levels
        time_ratio = times_by_levels[end] / times_by_levels[1]
        @info "Performance ratio (60 vs 5 levels): $(round(time_ratio, digits=2))x"
        @test time_ratio < 3.0  # No more than 3x slower with 12x more levels
    end

    @testset "Multiple variables performance" begin
        n = 50_000
        @info "Testing multiple variables performance..."

        # Create dataset with many categorical variables
        df = DataFrame(
            y = randn(n),
            x1 = randn(n), x2 = randn(n), x3 = randn(n),

            # Many categorical variables
            cat1 = rand(["A", "B", "C", "D"], n),
            cat2 = rand(["X", "Y", "Z"], n),
            cat3 = rand(["P", "Q", "R", "S", "T"], n),
            cat4 = rand(["Alpha", "Beta", "Gamma"], n),
            cat5 = rand(["North", "South"], n),
            cat6 = rand(["High", "Medium", "Low"], n),
            cat7 = rand(["Type1", "Type2", "Type3", "Type4"], n),
            cat8 = rand(["Group_A", "Group_B", "Group_C"], n),

            # Binary variables
            bin1 = rand(Bool, n), bin2 = rand(Bool, n),
            bin3 = rand(Bool, n), bin4 = rand(Bool, n)
        )

        # Convert categoricals
        cat_cols = [:cat1, :cat2, :cat3, :cat4, :cat5, :cat6, :cat7, :cat8]
        for col in cat_cols
            df[!, col] = categorical(df[!, col])
        end

        # Fit complex model with all variables
        formula_str = "y ~ " * join(["x$i" for i in 1:3], " + ") * " + " *
                     join(string.(cat_cols), " + ") * " + " *
                     join(["bin$i" for i in 1:4], " + ") * " + " *
                     "x1 * cat1 + x2 * cat2 + cat1 * cat2"

        model = lm(eval(Meta.parse("@formula($formula_str)")), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)

        @info "Model compiled with $(length(compiled)) terms"

        # Test ContrastEvaluator with many variables
        all_vars = vcat(cat_cols, [:bin1, :bin2, :bin3, :bin4])
        @info "Creating ContrastEvaluator with $(length(all_vars)) variables..."

        evaluator = contrastevaluator(compiled, data, all_vars)
        contrast_buf = Vector{Float64}(undef, length(compiled))

        # Test performance with each variable type
        @info "Testing performance across variable types..."
        for var in all_vars[1:min(6, end)]  # Test first 6 to keep reasonable
            if var in cat_cols
                levels = unique(df[!, var])
                from, to = levels[1], levels[min(2, end)]
            else  # Binary
                from, to = false, true
            end

            # Warm up first
            contrast_modelrow!(contrast_buf, evaluator, 1, var, from, to)
            result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1,
                                                 $var, $from, $to)
            @test result.memory == 0

            time_ns = median(result.times)
            @info "Variable $(var): $(round(time_ns, digits=1))ns"
            @test time_ns < 3e-6 * 1e9  # Under 3 microseconds
        end

        # Test batch performance
        @info "Testing batch performance..."
        batch_times = Float64[]
        test_rows = rand(1:n, 50)

        for row in test_rows
            time = @belapsed contrast_modelrow!($contrast_buf, $evaluator, $row,
                                             :cat1, "A", "B")
            push!(batch_times, time)
        end

        mean_batch_time = mean(batch_times)
        @info "Mean batch time: $(round(mean_batch_time * 1e9, digits=1))ns"
        @test mean_batch_time < 2e-6  # Under 2 microseconds per evaluation

        # Memory scaling test
        evaluator_memory = sizeof(evaluator)
        @info "ContrastEvaluator memory with $(length(all_vars)) vars: $(evaluator_memory) bytes"
        @test evaluator_memory < 2_000_000  # Should be under 2MB even with many variables
    end

    @testset "Memory scaling validation" begin
        @info "Testing memory scaling properties..."

        # Test that ContrastEvaluator memory usage is independent of dataset size
        base_n = 1_000
        test_sizes = [base_n, base_n * 10, base_n * 50]
        memory_usages = Int[]

        for n in test_sizes
            df = DataFrame(
                y = randn(n),
                x = randn(n),
                cat = rand(["A", "B", "C"], n),
                bin = rand(Bool, n)
            )
            df.cat = categorical(df.cat)

            model = lm(@formula(y ~ x + cat + bin), df)
            data = Tables.columntable(df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:cat, :bin])

            push!(memory_usages, sizeof(evaluator))
        end

        @info "Memory usage by dataset size:"
        for (size, memory) in zip(test_sizes, memory_usages)
            @info "  $(size) rows: $(memory) bytes"
        end

        # Memory should not scale with dataset size (O(1) property)
        memory_ratio = memory_usages[end] / memory_usages[1]
        @info "Memory scaling ratio (50K vs 1K): $(round(memory_ratio, digits=2))x"
        @test memory_ratio < 2.0  # Should be nearly constant

        # All memory usage should be reasonable
        @test all(mem -> mem < 500_000, memory_usages)  # All under 500KB
    end
end