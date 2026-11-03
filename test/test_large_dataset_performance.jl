# Large Dataset Performance Testing
# Test ContrastEvaluator performance and memory scaling with large datasets

using DataFrames, GLM, FormulaCompiler, Tables, CategoricalArrays, BenchmarkTools
using Test, Random

# Bytes allocated by one warm contrast_modelrow! call (function barrier, so no global access is counted)
function contrast_allocs(buf, evaluator, row, var, from, to)
    contrast_modelrow!(buf, evaluator, row, var, from, to)
    return @allocated contrast_modelrow!(buf, evaluator, row, var, from, to)
end

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

        # Zero allocations on random rows of the large dataset
        contrast_buf = Vector{Float64}(undef, length(compiled))
        for row in rand(1:n, 20)
            @test contrast_allocs(contrast_buf, evaluator, row, :treatment, "Control", "Treatment_A") == 0
            @test contrast_allocs(contrast_buf, evaluator, row, :region, "North", "South") == 0
        end
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
        @test contrast_allocs(contrast_buf, evaluator, 1, :high_cardinality, from_level, to_level) == 0

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

        # Zero allocations for every variable of the evaluator
        for var in all_vars
            if var in cat_cols
                levels = unique(df[!, var])
                from, to = levels[1], levels[2]
            else  # Binary
                from, to = false, true
            end
            @test contrast_allocs(contrast_buf, evaluator, 1, var, from, to) == 0
        end
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
            contrastevaluator(compiled, data, [:cat, :bin])   # warmup
            # Bytes allocated to build the evaluator (sizeof would give the constant struct size;
            # the evaluator references the data rather than copying it)
            push!(memory_usages, @allocated contrastevaluator(compiled, data, [:cat, :bin]))
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