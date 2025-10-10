# test_derivative_primitive_allocations.jl
# PRIMARY ALLOCATION TEST: Zero-allocation validation for derivative PRIMITIVES
#
# Purpose: Authoritative test for derivative_modelrow! allocation guarantees
# Scope:
#   - Single-row allocations (row 3)
#   - Multi-row allocations (rows 1, 50, 100, 150, 200, 250) - catches row-scaling bugs
#   - Both AD and FD backends
#   - Computational primitives only (statistical interface migrated to Margins.jl)
#
# Tests: ~12 total (derivative_modelrow! only)
#
# Note: Statistical interface tests (marginal_effects_eta!, marginal_effects_mu!,
#       delta_method_se) migrated to Margins/test/primitives/test_marginal_effects_allocations.jl
#
# julia --project="test" test/test_derivative_primitive_allocations.jl > test/test_derivative_primitive_allocations.txt 2>&1

using Test
using FormulaCompiler
using Tables
using DataFrames, Tables, GLM, CategoricalArrays
using BenchmarkTools

using FormulaCompiler: derivativeevaluator

"""
Non-capturing kernel for strict zero-allocation BenchmarkTools checks
"""
function _bench_derivative_modelrow!(Jloc, deloc, rowloc)
    derivative_modelrow!(Jloc, deloc, rowloc)
    return nothing
end

@testset "Derivative Primitive Allocation Checks" begin
    results = DataFrame(
        path = String[],
        backend = String[],
        min_memory_bytes = Int[],
        min_time_seconds = Float64[],
    )

    # Test setup
    n = 300
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group3 = categorical(rand(["A", "B", "C"], n)),
    )
    data = Tables.columntable(df)
    model = lm(@formula(y ~ 1 + x + z + x & group3), df)
    compiled = compile_formula(model, data)
    vars = [:x, :z]
    β = coef(model)

    # Build derivative evaluators (concrete types)
    de_ad = derivativeevaluator(:ad, compiled, data, vars)
    de_fd = derivativeevaluator(:fd, compiled, data, vars)

    # Buffers for derivative tests
    J = Matrix{Float64}(undef, length(compiled), length(vars))  # Jacobian matrix
    row_vec = Vector{Float64}(undef, length(compiled))

    # Warmup
    compiled(row_vec, data, 2)
    derivative_modelrow!(J, de_ad, 2)
    derivative_modelrow!(J, de_fd, 2)

    # Core compiled evaluation: expect 0 allocations
    b_comp = @benchmark $compiled($row_vec, $data, 3) samples=600
    push!(results, ("compiled_row", "base", minimum(b_comp.memory), minimum(b_comp.times)))
    @test results[end, :min_memory_bytes] == 0

    # === derivative_modelrow! (core Jacobian evaluation) ===

    # AD backend: zero allocations required
    b_derivative_ad = @benchmark derivative_modelrow!($J, $de_ad, 3) samples=400
    push!(results, ("derivative_modelrow", "ad", minimum(b_derivative_ad.memory), minimum(b_derivative_ad.times)))
    @test results[end, :min_memory_bytes] == 0

    # FD backend: zero allocations
    b_derivative_fd = @benchmark derivative_modelrow!($J, $de_fd, 3) samples=400
    push!(results, ("derivative_modelrow", "fd", minimum(b_derivative_fd.memory), minimum(b_derivative_fd.times)))
    @test results[end, :min_memory_bytes] == 0

    b_derivative_fd_strict = @benchmark _bench_derivative_modelrow!($J, $de_fd, 3) samples=400
    @test minimum(b_derivative_fd_strict.memory) == 0

    # === Multi-row allocation validation (row-scaling check) ===

    # Test allocations across different rows to catch row-dependent allocations
    # This validates that categorical reference index optimization eliminates row-scaling allocations
    test_rows = [1, 50, 100, 150, 200, 250]

    @testset "Multi-row allocation validation (primitives)" begin
        for test_row in test_rows
            # AD backend allocations
            b_ad = @benchmark derivative_modelrow!($J, $de_ad, $test_row) samples=100 evals=1
            @test minimum(b_ad).memory == 0

            # FD backend allocations
            b_fd = @benchmark derivative_modelrow!($J, $de_fd, $test_row) samples=100 evals=1
            @test minimum(b_fd).memory == 0
        end
    end

    # === Cross-validation: AD vs FD backends ===

    # Verify both backends produce consistent Jacobian results (mathematical correctness)
    J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
    J_fd = Matrix{Float64}(undef, length(compiled), length(vars))

    # Test multiple rows for robustness
    for test_row in [1, 50, 150, 299]
        # Jacobian comparison
        derivative_modelrow!(J_ad, de_ad, test_row)
        derivative_modelrow!(J_fd, de_fd, test_row)

        @test J_ad ≈ J_fd rtol=1e-6 atol=1e-8
    end
end
