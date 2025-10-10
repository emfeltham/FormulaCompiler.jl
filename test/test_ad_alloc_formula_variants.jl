# test_ad_alloc_formula_variants.jl
# Tests AD allocation characteristics across diverse formula patterns
#
# Purpose: Validate that AD backend maintains zero-allocation guarantees for standard formulas
#          and identify allocation sources in formulas with boolean predicates
# Scope: Tests continuous, categorical, interactions, transforms, and boolean predicates
# Expected: 0 allocations for standard formulas; small allocations for boolean predicates
#
# Run with: julia --project=. -e 'using Pkg; Pkg.test("FormulaCompiler")'
# Or standalone: julia --project=. test/test_ad_alloc_formula_variants.jl

using Test
using BenchmarkTools
using GLM
using DataFrames
using Tables
using FormulaCompiler

# Load data generator (only when running standalone)
if !@isdefined(generate_synthetic_dataset)
    include("support/generate_large_synthetic_data.jl")
end

@testset "AD Allocation Formula Variants" begin
    # Generate test dataset
    n_test = 1000
    df = generate_synthetic_dataset(n_test; seed=08540)

    # Helper function to test a formula variant
    function test_formula_allocations(formula, vars_to_test, expect_zero::Bool)
        @debug "Testing formula" formula vars_to_test

        # Fit model
        model = glm(formula, df, Bernoulli(), LogitLink())

        # Compile with copied data (to eliminate NamedTuple size issue)
        data_cols_original = Tables.columntable(df)
        data_cols = NamedTuple{keys(data_cols_original)}(map(copy, values(data_cols_original)))
        compiled = compile_formula(model, data_cols)

        # Build derivative evaluator
        de_ad = derivativeevaluator(:ad, compiled, data_cols, vars_to_test)
        J = Matrix{Float64}(undef, length(compiled), length(vars_to_test))

        # Warmup
        derivative_modelrow!(J, de_ad, 1)

        # Benchmark
        b = @benchmark derivative_modelrow!($J, $de_ad, 1) samples=400

        min_allocs = minimum(b).allocs
        min_bytes = minimum(b).memory

        @debug "Allocation results" min_allocs min_bytes expect_zero

        return min_allocs, min_bytes
    end

    @testset "Continuous Variables Only" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~ dists_p_inv + are_related_dists_a_inv),
            [:dists_p_inv, :are_related_dists_a_inv],
            true
        )
        @test allocs == 0
        @test bytes == 0
    end

    @testset "Boolean Variable (Not Predicate)" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~ socio4 + dists_p_inv + are_related_dists_a_inv),
            [:dists_p_inv, :are_related_dists_a_inv],
            true
        )
        @test allocs == 0
        @test bytes == 0
    end

    @testset "Boolean × Continuous Interaction" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~ socio4 + dists_p_inv + socio4 & dists_p_inv),
            [:dists_p_inv],
            true
        )
        @test allocs == 0
        @test bytes == 0
    end

    @testset "Boolean Predicate (Comparison)" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~ dists_p_inv + (dists_a_inv <= inv(2))),
            [:dists_p_inv],
            false  # Expect allocations from boolean predicate
        )
        # Boolean predicates
        @test allocs == 0  # boolean predicates allocate
        @test bytes == 0
        @debug "Boolean predicate allocations (expected)" allocs bytes
    end

    @testset "Categorical Variable" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~ dists_p_inv + relation),
            [:dists_p_inv],
            true
        )
        @test allocs == 0
        @test bytes == 0
    end

    @testset "Categorical × Continuous Interaction" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~ dists_p_inv + dists_p_inv & relation),
            [:dists_p_inv],
            true
        )
        @test allocs == 0
        @test bytes == 0
    end

    @testset "Complex Categorical Interactions" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~
                socio4 +
                (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
                (schoolyears_p + wealth_d1_4_p + relation) & (1 + socio4)
            ),
            [:dists_p_inv, :are_related_dists_a_inv, :schoolyears_p, :wealth_d1_4_p],
            true
        )
        @test allocs == 0
        @test bytes == 0
    end

    @testset "Full Complex Formula (No Boolean Predicate)" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~
                socio4 +
                (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
                !socio4 & dists_a_inv +
                (schoolyears_p + wealth_d1_4_p + man_p + age_p + religion_c_p +
                same_building + population +
                hhi_religion + hhi_indigenous +
                coffee_cultivation + market +
                relation) & (1 + socio4 + are_related_dists_a_inv) +
                (
                    degree_a_mean + degree_h +
                    age_a_mean + age_h * age_h_nb_1_socio +
                    schoolyears_a_mean + schoolyears_h * schoolyears_h_nb_1_socio +
                    man_x * man_x_mixed_nb_1 +
                    wealth_d1_4_a_mean + wealth_d1_4_h * wealth_d1_4_h_nb_1_socio +
                    isindigenous_x * isindigenous_homop_nb_1 +
                    religion_c_x * religion_homop_nb_1
                ) & (1 + socio4 + are_related_dists_a_inv) +
                religion_c_x & hhi_religion +
                isindigenous_x & hhi_indigenous
            ),
            [:age_h, :dists_p_inv, :are_related_dists_a_inv, :schoolyears_h],
            true
        )
        @test allocs == 0
        @test bytes == 0
    end

    @testset "Full Complex Formula (With Boolean Predicate)" begin
        allocs, bytes = test_formula_allocations(
            @formula(response ~
                socio4 +
                (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
                !socio4 & dists_a_inv +
                (num_common_nbs & (dists_a_inv <= inv(2))) & (1 + are_related_dists_a_inv + dists_p_inv) +
                (schoolyears_p + wealth_d1_4_p + man_p + age_p + religion_c_p +
                same_building + population +
                hhi_religion + hhi_indigenous +
                coffee_cultivation + market +
                relation) & (1 + socio4 + are_related_dists_a_inv) +
                (
                    degree_a_mean + degree_h +
                    age_a_mean + age_h * age_h_nb_1_socio +
                    schoolyears_a_mean + schoolyears_h * schoolyears_h_nb_1_socio +
                    man_x * man_x_mixed_nb_1 +
                    wealth_d1_4_a_mean + wealth_d1_4_h * wealth_d1_4_h_nb_1_socio +
                    isindigenous_x * isindigenous_homop_nb_1 +
                    religion_c_x * religion_homop_nb_1
                ) & (1 + socio4 + are_related_dists_a_inv) +
                religion_c_x & hhi_religion +
                isindigenous_x & hhi_indigenous
            ),
            [:age_h, :dists_p_inv, :are_related_dists_a_inv, :schoolyears_h],
            false  # Expect allocations from boolean predicate
        )
        @test allocs == 0
        @test bytes == 0
        @debug "Complex formula with boolean predicate" allocs bytes
    end
end
