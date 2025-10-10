# test_formulacompiler_primitives_allocations.jl
# BATCH SCALING & REGRESSION GUARDS: Core primitive allocation validation
#
# Purpose: Tests FormulaCompiler primitives with complex formulas and batch operations
# Scope:
#   - Core primitives: compiled(), modelrow!(), derivative_modelrow!()
#   - Complex formula patterns (interactions, transforms, categoricals)
#   - Batch scaling tests (1, 10, 100, 1000 rows)
#   - NamedTuple size regression guards
#
# Unique value: Validates allocation stability under batch operations
# Complements: test_derivative_allocations.jl (which focuses on single/multi-row patterns)
#
# EXPECTED BEHAVIOR:
# - compiled(): 0 allocations (direct formula evaluation)
# - derivative_modelrow!() with FD backend: 0 allocations for ALL cases
# - derivative_modelrow!() with AD backend: 0 allocations with proper data_cols
#
# Wide NamedTuple handling:
# Prior versions leaked ~16 bytes per variable when the baseline NamedTuple exceeded
# ~50 columns. The ADEvaluator refactor (2025-10-04) introduced cached column tuples
# and explicit row setters so even wide tables hit 0 allocations. These tests retain
# the wide dataset to guard against regressions. This was not driven by `jacobian!`
# which was zero-allocation.
#
# Run with:
# julia --project=. test/test_formulacompiler_primitives_allocations.jl  > test/test_formulacompiler_primitives_allocations.txt 2>&1

using Test
using BenchmarkTools
using GLM
using DataFrames
using Tables
using FormulaCompiler
using FormulaCompiler: compile_formula, modelrow!, derivative_modelrow!

# Load data generator (only when running standalone)
if !@isdefined(generate_synthetic_dataset)
    include("support/generate_large_synthetic_data.jl")
end

function batch_derivatives(de, J, n_rows)
    for i in 1:n_rows
        derivative_modelrow!(J, de, i)
    end
end

# Test function that wraps modelrow() for allocation testing
function test_modelrow(compiled, data_cols, row_idx, output)
    modelrow!(output, compiled, data_cols, row_idx)
end

function test_derivative(de, J, row_idx)
    derivative_modelrow!(J, de, row_idx)
end

function batch_modelrow(compiled, data_cols, n_rows, output)
    for i in 1:n_rows
        modelrow!(output, compiled, data_cols, i)
    end
end

n_test = 500_000

@testset "FormulaCompiler Primitives - Complex Formula Allocations" begin
    # Generate test dataset (smaller for faster testing)
    df = generate_synthetic_dataset(n_test; seed=08540)

    # Fit the complex model from complex_example.jl
    fx = @formula(response ~
        socio4 +
        (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
        !socio4 & dists_a_inv +
        # (num_common_nbs & (dists_a_inv <= inv(2))) & (1 + are_related_dists_a_inv + dists_p_inv) +
        # individual variables
        (schoolyears_p + wealth_d1_4_p + man_p + age_p + religion_c_p +
        same_building + population +
        hhi_religion + hhi_indigenous +
        coffee_cultivation + market +
        relation) & (1 + socio4 + are_related_dists_a_inv) +
        # tie variables
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
    )

    model = fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())

    # Compile formula
    data_cols_original = Tables.columntable(df)

    # test allocations by data access:
    # julia --project=. scripts/test_data_cols_modification.jl
    # test_data_cols_modification: Test option 2: Copy all columns to break any views/references
    data_cols = NamedTuple{keys(data_cols_original)}(map(copy, values(data_cols_original)))

    compiled = compile_formula(model, data_cols)

    # Test continuous variables for derivatives
    continuous_vars = [:age_h, :dists_p_inv, :are_related_dists_a_inv, :schoolyears_h]

    @testset "modelrow() - Direct formula evaluation" begin
        # Pre-allocate output buffer
        output = Vector{Float64}(undef, length(compiled))

        # Warmup
        test_modelrow(compiled, data_cols, 1, output)

        # Measure allocations for single row evaluation
        bench = @benchmark $test_modelrow($compiled, $data_cols, 1, $output) samples=1000 evals=1

        min_allocs = minimum(bench).allocs
        min_bytes = minimum(bench).memory
        median_time_ns = median(bench).time

        println("\n=== modelrow() Allocations ===")
        println("  Allocations: $min_allocs")
        println("  Memory: $min_bytes bytes")
        println("  Median time: $(median_time_ns) ns")

        @test min_allocs == 0
        @test min_bytes == 0
    end

    @testset "derivative_modelrow!() - AD backend" begin
        # Build derivative evaluator
        de = derivativeevaluator(:ad, compiled, data_cols, continuous_vars)

        # Pre-allocate Jacobian buffer (matrix: n_terms × n_vars)
        J = Matrix{Float64}(undef, length(compiled), length(continuous_vars))

        # Warmup
        test_derivative(de, J, 1)

        # Measure allocations
        bench = @benchmark $test_derivative($de, $J, 1) samples=1000 evals=1

        min_allocs = minimum(bench).allocs
        min_bytes = minimum(bench).memory
        median_time_ns = median(bench).time

        println("\n=== derivative_modelrow!() Allocations (AD backend) ===")
        println("  Variables: $continuous_vars")
        println("  Allocations: $min_allocs")
        println("  Memory: $min_bytes bytes")
        println("  Median time: $(median_time_ns) ns")
        println("  NOTE: Uses 53-column NamedTuple; cached columns keep allocations at zero")

        @test min_allocs == 0
        @test min_bytes == 0
    end

    @testset "derivative_modelrow!() - FD backend" begin
        # Build derivative evaluator for finite differences backend
        de = derivativeevaluator(:fd, compiled, data_cols, continuous_vars)

        # Pre-allocate Jacobian buffer
        J = Matrix{Float64}(undef, length(compiled), length(continuous_vars))

        # Warmup
        test_derivative(de, J, 1)

        # Measure allocations
        bench = @benchmark $test_derivative($de, $J, 1) samples=1000 evals=1

        min_allocs = minimum(bench).allocs
        min_bytes = minimum(bench).memory
        median_time_ns = median(bench).time

        println("\n=== derivative_modelrow!() Allocations (FD backend) ===")
        println("  Variables: $continuous_vars")
        println("  Allocations: $min_allocs")
        println("  Memory: $min_bytes bytes")
        println("  Median time: $(median_time_ns) ns")

        @test min_allocs == 0
        @test min_bytes == 0
    end

    @testset "Batch evaluation scaling test" begin
        # Test that allocations remain zero across multiple rows
        output = Vector{Float64}(undef, length(compiled))

        # Warmup
        batch_modelrow(compiled, data_cols, 10, output)

        # Test with different batch sizes
        for batch_size in [100, 1000, n_test]
            bench = @benchmark $batch_modelrow($compiled, $data_cols, $batch_size, $output) samples=10 evals=1

            min_allocs = minimum(bench).allocs
            allocs_per_row = min_allocs / batch_size

            println("\n=== Batch modelrow() n=$batch_size ===")
            println("  Total allocations: $min_allocs")
            println("  Allocations/row: $(round(allocs_per_row, digits=6))")

            @test allocs_per_row <= 0.0
        end
    end

    @testset "Batch derivative scaling test - AD" begin
        de = derivativeevaluator(:ad, compiled, data_cols, continuous_vars)
        J = Matrix{Float64}(undef, length(compiled), length(continuous_vars))

        # Warmup
        batch_derivatives(de, J, 10)

        for batch_size in [100, 1000, n_test]
            bench = @benchmark $batch_derivatives($de, $J, $batch_size) samples=10 evals=1

            min_allocs = minimum(bench).allocs
            allocs_per_row = min_allocs / batch_size

            println("\n=== Batch derivative_modelrow!() AD n=$batch_size ===")
            println("  Total allocations: $min_allocs")
            println("  Allocations/row: $(round(allocs_per_row, digits=6)) (target: 0)")

            @test min_allocs == 0
            @test allocs_per_row == 0.0
        end
    end

    @testset "Batch derivative scaling test - FD" begin
        # Test that derivative allocations remain zero across multiple rows
        de = derivativeevaluator(:fd, compiled, data_cols, continuous_vars)
        J = Matrix{Float64}(undef, length(compiled), length(continuous_vars))

        # Warmup
        batch_derivatives(de, J, 10)

        # Test with different batch sizes
        for batch_size in [100, 1000, n_test]
            bench = @benchmark $batch_derivatives($de, $J, $batch_size) samples=10 evals=1

            min_allocs = minimum(bench).allocs
            allocs_per_row = min_allocs / batch_size

            println("\n=== Batch derivative_modelrow!() FD n=$batch_size ===")
            println("  Total allocations: $min_allocs")
            println("  Allocations/row: $(round(allocs_per_row, digits=6))")

            @test allocs_per_row <= 0.0
        end
    end

    @testset "NamedTuple size regression guard - AD backend" begin
        println("\n=== NamedTuple Size Impact on AD Allocations ===")

        de = derivativeevaluator(:ad, compiled, data_cols, continuous_vars)
        J = Matrix{Float64}(undef, length(compiled), length(continuous_vars))

        bench = @benchmark $test_derivative($de, $J, 1) samples=1000 evals=1

        min_allocs = minimum(bench).allocs
        min_bytes = minimum(bench).memory

        println("  Allocations: $min_allocs")
        println("  Memory: $min_bytes bytes")
        println("  Dataset columns: $(length(keys(data_cols)))")

        @test min_allocs == 0
        @test min_bytes == 0
    end
end

