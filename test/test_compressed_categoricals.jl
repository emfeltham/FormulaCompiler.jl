# test_compressed_categoricals.jl
# Tests for compressed categorical arrays (UInt8, UInt16 reference types)
#
# PURPOSE: Ensure FormulaCompiler handles all categorical reference types correctly
# COVERAGE: UInt8 (compress=true, <256 levels), UInt16 (256-65535 levels), UInt32 (default)
# REGRESSION: Catches bugs that only appear with non-UInt32 reference types
#
# Historical context: Bugs in continuous_variables() and contrastevaluator() went
# undetected because all FC tests used default UInt32 categoricals. Real-world datasets
# (like RDatasets cbpp) use compressed categoricals, exposing type mismatch issues.

using Test
using FormulaCompiler
using CategoricalArrays
using GLM
using DataFrames
using Tables

@testset "Compressed categorical arrays" begin

    @testset "UInt8 compressed categoricals (< 256 levels)" begin
        # Create compressed categorical with 4 levels → UInt8 refs
        n = 100
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            cat_uint8 = categorical(rand(["A", "B", "C", "D"], n), compress=true)
        )

        # Verify it's actually UInt8
        @test eltype(df.cat_uint8.refs) == UInt8

        # Test basic compilation
        data = Tables.columntable(df)
        model = lm(@formula(y ~ x + cat_uint8), df)
        compiled = compile_formula(model, data)

        # Test evaluation
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, data, 1)
        @test length(output) == length(compiled)
        @test all(isfinite, output)

        # Test continuous_variables - should NOT include cat_uint8
        cont_vars = continuous_variables(compiled, data)
        @test :x ∈ cont_vars
        @test :cat_uint8 ∉ cont_vars

        # Test ContrastEvaluator with UInt8 categorical
        contrast_eval = contrastevaluator(compiled, data, [:cat_uint8])
        contrast_buf = Vector{Float64}(undef, length(compiled))

        # Test contrast evaluation with UInt8 categorical
        contrast_modelrow!(contrast_buf, contrast_eval, 1, :cat_uint8, "A", "B")
        @test all(isfinite, contrast_buf)

        # Verify level map has correct UInt8 type
        level_map = contrast_eval.categorical_level_maps[1]
        @test level_map.levels[1][2] isa UInt8  # Reference index should be UInt8
        @test level_map.levels[2][2] isa UInt8
    end

    @testset "UInt16 compressed categoricals (256-65535 levels)" begin
        # Create categorical with 300 levels → UInt16 refs with compress
        # IMPORTANT: Must ensure all levels are present in data, otherwise CategoricalArrays
        # will compress to smaller ref type. Use at least one observation per level.
        levels_300 = string.(1:300)
        # Create data with all 300 levels present (at least once each)
        n = 300  # One of each level
        level_data = levels_300  # All levels present once
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            cat_uint16 = categorical(level_data, levels=levels_300, compress=true)
        )

        # Verify it's actually UInt16 (requires >255 ACTUAL levels in data)
        @test eltype(df.cat_uint16.refs) == UInt16

        # Test basic compilation
        data = Tables.columntable(df)
        model = lm(@formula(y ~ x + cat_uint16), df)
        compiled = compile_formula(model, data)

        # Test evaluation
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, data, 1)
        @test all(isfinite, output)

        # Test continuous_variables - should NOT include cat_uint16
        cont_vars = continuous_variables(compiled, data)
        @test :x ∈ cont_vars
        @test :cat_uint16 ∉ cont_vars

        # Test ContrastEvaluator with UInt16 categorical
        contrast_eval = contrastevaluator(compiled, data, [:cat_uint16])
        contrast_buf = Vector{Float64}(undef, length(compiled))

        # Test contrast evaluation with UInt16 categorical
        contrast_modelrow!(contrast_buf, contrast_eval, 1, :cat_uint16, "1", "2")
        @test all(isfinite, contrast_buf)

        # Verify level map has correct UInt16 type
        level_map = contrast_eval.categorical_level_maps[1]
        @test level_map.levels[1][2] isa UInt16
        @test level_map.levels[2][2] isa UInt16
    end

    @testset "UInt32 default categoricals (regression check)" begin
        # Ensure we didn't break default UInt32 behavior
        n = 100
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            cat_uint32 = categorical(rand(["Control", "Treatment"], n))  # No compress
        )

        # Verify it's UInt32 (default)
        @test eltype(df.cat_uint32.refs) == UInt32

        # Test basic compilation
        data = Tables.columntable(df)
        model = lm(@formula(y ~ x + cat_uint32), df)
        compiled = compile_formula(model, data)

        # Test continuous_variables
        cont_vars = continuous_variables(compiled, data)
        @test :x ∈ cont_vars
        @test :cat_uint32 ∉ cont_vars

        # Test ContrastEvaluator
        contrast_eval = contrastevaluator(compiled, data, [:cat_uint32])
        contrast_buf = Vector{Float64}(undef, length(compiled))
        contrast_modelrow!(contrast_buf, contrast_eval, 1, :cat_uint32, "Control", "Treatment")
        @test all(isfinite, contrast_buf)

        # Verify level map has correct UInt32 type
        level_map = contrast_eval.categorical_level_maps[1]
        @test level_map.levels[1][2] isa UInt32
    end

    @testset "continuous_variables with counterfactual data" begin
        # Test that continuous_variables correctly filters categoricals
        # when data contains CategoricalCounterfactualVector
        n = 50
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            cat_compressed = categorical(rand(["A", "B", "C"], n), compress=true)
        )

        data = Tables.columntable(df)
        model = lm(@formula(y ~ x + cat_compressed), df)
        compiled = compile_formula(model, data)

        # Build counterfactual data (wraps categorical in CategoricalCounterfactualVector)
        data_cf, counterfactuals = FormulaCompiler.build_counterfactual_data(
            data, [:cat_compressed], 1
        )

        # Verify we have a counterfactual vector
        @test data_cf.cat_compressed isa FormulaCompiler.CategoricalCounterfactualVector
        @test eltype(data_cf.cat_compressed) == UInt8  # Should be UInt8, not CategoricalValue

        # Test continuous_variables with counterfactual-wrapped data
        # BUG: This incorrectly classified cat_compressed as continuous before fix
        # because eltype(CategoricalCounterfactualVector{String,UInt8}) == UInt8 <: Real
        cont_vars = continuous_variables(compiled, data_cf)
        @test :x ∈ cont_vars
        @test :cat_compressed ∉ cont_vars  # Should be filtered out!
    end

    @testset "Mixed reference types in same model" begin
        # Test model with multiple categorical variables of different ref types
        levels_300 = string.(1:300)
        n = 300  # Ensure all levels present
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            small_cat = categorical(rand(["A", "B"], n), compress=true),                   # UInt8
            large_cat = categorical(levels_300, levels=levels_300, compress=true),          # UInt16
            default_cat = categorical(rand(["X", "Y", "Z"], n))                             # UInt32
        )

        @test eltype(df.small_cat.refs) == UInt8
        @test eltype(df.large_cat.refs) == UInt16
        @test eltype(df.default_cat.refs) == UInt32

        # Compile with all three
        data = Tables.columntable(df)
        model = lm(@formula(y ~ x + small_cat + large_cat + default_cat), df)
        compiled = compile_formula(model, data)

        # Test continuous_variables
        cont_vars = continuous_variables(compiled, data)
        @test :x ∈ cont_vars
        @test :small_cat ∉ cont_vars
        @test :large_cat ∉ cont_vars
        @test :default_cat ∉ cont_vars

        # Test ContrastEvaluator with multiple ref types
        contrast_eval = contrastevaluator(compiled, data, [:small_cat, :large_cat, :default_cat])

        # Verify each level map has correct type
        @test contrast_eval.categorical_level_maps[1].levels[1][2] isa UInt8
        @test contrast_eval.categorical_level_maps[2].levels[1][2] isa UInt16
        @test contrast_eval.categorical_level_maps[3].levels[1][2] isa UInt32

        # Test contrasts for each
        contrast_buf = Vector{Float64}(undef, length(compiled))
        contrast_modelrow!(contrast_buf, contrast_eval, 1, :small_cat, "A", "B")
        @test all(isfinite, contrast_buf)

        contrast_modelrow!(contrast_buf, contrast_eval, 1, :large_cat, "1", "2")
        @test all(isfinite, contrast_buf)

        contrast_modelrow!(contrast_buf, contrast_eval, 1, :default_cat, "X", "Y")
        @test all(isfinite, contrast_buf)
    end
end

println("✓ All compressed categorical tests passed")
