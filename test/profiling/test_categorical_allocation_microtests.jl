# test_categorical_allocation_microtests.jl
# Focused BenchmarkTools-based allocation microtests for categorical-heavy cases
# julia --project="." test/test_categorical_allocation_microtests.jl > test/test_categorical_allocation_microtests.txt 2>&1

using Test
using Random
using BenchmarkTools
using DataFrames, Tables, CategoricalArrays
using GLM, StatsModels
using FormulaCompiler

# Helper: assert zero allocations for a compiled evaluator call
function _assert_zero_allocs(compiled, output, data, row)
    # Warmup
    compiled(output, data, row)
    # Benchmark with interpolation to avoid closure allocations
    b = @benchmark $compiled($output, $data, $row) samples=300 evals=1
    @test minimum(b.memory) == 0
    return b
end

n = 10_000_000

@testset "Categorical Allocation Microtests" begin
    Random.seed!(0xCA71)

    @testset "Single categorical (10 levels)" begin
        
        levels10 = string.('A':'J')
        df = DataFrame(
            y = randn(n),
            group10 = categorical(rand(levels10, n), levels=levels10),
        )
        data = Tables.columntable(df)
        model = lm(@formula(y ~ group10), df)
        compiled = compile_formula(model, data)
        output = Vector{Float64}(undef, length(compiled))
        _ = _assert_zero_allocs(compiled, output, data, 1)
    end

    @testset "x * group (5 levels)" begin
        
        levels5 = ["P","Q","R","S","T"]
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            group5 = categorical(rand(levels5, n), levels=levels5),
        )
        data = Tables.columntable(df)
        model = lm(@formula(y ~ x * group5), df)
        compiled = compile_formula(model, data)
        output = Vector{Float64}(undef, length(compiled))
        _ = _assert_zero_allocs(compiled, output, data, 2)
    end

    @testset "groupA * groupB (categorical × categorical)" begin

        a_levels = ["W","X","Y","Z"]
        b_levels = ["L","M","N"]
        df = DataFrame(
            y = randn(n),
            groupA = categorical(rand(a_levels, n), levels=a_levels),
            groupB = categorical(rand(b_levels, n), levels=b_levels),
        )
        data = Tables.columntable(df)
        model = lm(@formula(y ~ groupA * groupB), df)
        compiled = compile_formula(model, data)
        output = Vector{Float64}(undef, length(compiled))
        _ = _assert_zero_allocs(compiled, output, data, 3)
    end

    @testset "Boolean categorical (Vector{Bool})" begin
        
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            flag = rand(Bool, n),
        )
        data = Tables.columntable(df)

        # As plain Bool column
        model_bool = lm(@formula(y ~ flag), df)
        compiled_bool = compile_formula(model_bool, data)
        out_bool = Vector{Float64}(undef, length(compiled_bool))
        _ = _assert_zero_allocs(compiled_bool, out_bool, data, 4)

        # Interaction with continuous
        model_inter = lm(@formula(y ~ x * flag), df)
        compiled_inter = compile_formula(model_inter, data)
        out_inter = Vector{Float64}(undef, length(compiled_inter))
        _ = _assert_zero_allocs(compiled_inter, out_inter, data, 5)
    end

    @testset "Categorical{Bool} (CategoricalArray{Bool})" begin
        
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            flag_cat = categorical(rand(Bool, n)),
        )
        data = Tables.columntable(df)

        # As categorical Bool
        model_cat = lm(@formula(y ~ flag_cat), df)
        compiled_cat = compile_formula(model_cat, data)
        out_cat = Vector{Float64}(undef, length(compiled_cat))
        _ = _assert_zero_allocs(compiled_cat, out_cat, data, 6)

        # Interaction with continuous
        model_cat_inter = lm(@formula(y ~ x * flag_cat), df)
        compiled_cat_inter = compile_formula(model_cat_inter, data)
        out_cat_inter = Vector{Float64}(undef, length(compiled_cat_inter))
        _ = _assert_zero_allocs(compiled_cat_inter, out_cat_inter, data, 7)
    end
end
