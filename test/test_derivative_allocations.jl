# test_derivative_allocations.jl
# PRIMARY ALLOCATION TEST: Comprehensive zero-allocation validation for derivatives
#
# Purpose: Authoritative test for derivative allocation guarantees
# Scope:
#   - Single-row allocations (row 3)
#   - Multi-row allocations (rows 1, 50, 100, 150, 200, 250) - catches row-scaling bugs
#   - Tight loop patterns (50K iterations) - validates AME usage
#   - Both AD and FD backends
#   - All link functions (Identity, Log, Logit, Probit)
#   - Parameter gradient matrices (Gβ)
#   - Delta method standard errors
#
# Tests: 69 total (33 single-row + 36 multi-row)
#
# julia --project="test" test/test_derivative_allocations.jl > test/test_derivative_allocations.txt 2>&1

using Test
using FormulaCompiler
using Tables
using DataFrames, Tables, GLM, CategoricalArrays
using BenchmarkTools

using FormulaCompiler: derivativeevaluator

"""
Non-capturing kernels for strict zero-allocation BenchmarkTools checks
"""
function _bench_derivative_modelrow!(Jloc, deloc, rowloc)
    derivative_modelrow!(Jloc, deloc, rowloc)
    return nothing
end


function _bench_eta_marginal_effects_with_grad!(gloc, Gβloc, deloc, bloc, rowloc)
    marginal_effects_eta!(gloc, Gβloc, deloc, bloc, rowloc)
    return nothing
end

function _bench_mu_marginal_effects_with_grad!(gloc, Gβloc, deloc, bloc, linkloc, rowloc)
    marginal_effects_mu!(gloc, Gβloc, deloc, bloc, linkloc, rowloc)
    return nothing
end

@testset "Parameter Gradient Matrix Allocation Checks" begin
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

    # Buffers for Phase 1 functions
    g = Vector{Float64}(undef, length(vars))          # Marginal effects vector
    Gβ = Matrix{Float64}(undef, length(compiled), length(vars))  # Parameter gradient matrix
    row_vec = Vector{Float64}(undef, length(compiled))

    # Warmup all functions
    compiled(row_vec, data, 2)
    marginal_effects_eta!(g, Gβ, de_ad, β, 2)
    marginal_effects_eta!(g, Gβ, de_fd, β, 2)
    marginal_effects_mu!(g, Gβ, de_ad, β, LogitLink(), 2)
    marginal_effects_mu!(g, Gβ, de_fd, β, LogitLink(), 2)

    # Core compiled evaluation: expect 0 allocations
    b_comp = @benchmark $compiled($row_vec, $data, 3) samples=600
    push!(results, ("compiled_row", "base", minimum(b_comp.memory), minimum(b_comp.times)))
    @test results[end, :min_memory_bytes] == 0

    # === derivative_modelrow! (core Jacobian evaluation) ===

    # AD backend: zero allocations required
    b_derivative_ad = @benchmark derivative_modelrow!($Gβ, $de_ad, 3) samples=400
    push!(results, ("derivative_modelrow", "ad", minimum(b_derivative_ad.memory), minimum(b_derivative_ad.times)))
    @test results[end, :min_memory_bytes] == 0

    # FD backend: zero allocations
    b_derivative_fd = @benchmark derivative_modelrow!($Gβ, $de_fd, 3) samples=400
    push!(results, ("derivative_modelrow", "fd", minimum(b_derivative_fd.memory), minimum(b_derivative_fd.times)))
    @test results[end, :min_memory_bytes] == 0

    b_derivative_fd_strict = @benchmark _bench_derivative_modelrow!($Gβ, $de_fd, 3) samples=400
    @test minimum(b_derivative_fd_strict.memory) == 0

    # === η-scale marginal effects with parameter gradients ===

    # AD backend: zero allocations required
    b_eta_grad_ad = @benchmark marginal_effects_eta!($g, $Gβ, $de_ad, $β, 3) samples=400
    push!(results, ("eta_marginal_effects_with_grad", "ad", minimum(b_eta_grad_ad.memory), minimum(b_eta_grad_ad.times)))
    @test results[end, :min_memory_bytes] == 0

    # FD backend: zero allocations
    b_eta_grad_fd = @benchmark marginal_effects_eta!($g, $Gβ, $de_fd, $β, 3) samples=400
    push!(results, ("eta_marginal_effects_with_grad", "fd", minimum(b_eta_grad_fd.memory), minimum(b_eta_grad_fd.times)))
    @test results[end, :min_memory_bytes] == 0

    # Strict zero-allocation check using non-capturing kernel (FD only)
    b_eta_grad_fd_strict = @benchmark _bench_eta_marginal_effects_with_grad!($g, $Gβ, $de_fd, $β, 3) samples=400
    @test minimum(b_eta_grad_fd_strict.memory) == 0

    # === μ-scale marginal effects with parameter gradients ===

    # AD backend with Logit link: zero allocations required
    b_mu_grad_ad = @benchmark marginal_effects_mu!($g, $Gβ, $de_ad, $β, LogitLink(), 3) samples=400
    push!(results, ("mu_marginal_effects_with_grad_logit", "ad", minimum(b_mu_grad_ad.memory), minimum(b_mu_grad_ad.times)))
    @test results[end, :min_memory_bytes] == 0

    # FD backend with Logit link: zero allocations
    b_mu_grad_fd = @benchmark marginal_effects_mu!($g, $Gβ, $de_fd, $β, LogitLink(), 3) samples=400
    push!(results, ("mu_marginal_effects_with_grad_logit", "fd", minimum(b_mu_grad_fd.memory), minimum(b_mu_grad_fd.times)))
    @test results[end, :min_memory_bytes] == 0

    # Strict zero-allocation check using non-capturing kernel (FD only)
    b_mu_grad_fd_strict = @benchmark _bench_mu_marginal_effects_with_grad!($g, $Gβ, $de_fd, $β, LogitLink(), 3) samples=400
    @test minimum(b_mu_grad_fd_strict.memory) == 0

    # Test other link functions for μ-scale with parameter gradients
    for (link_name, link) in [("identity", IdentityLink()), ("log", LogLink()), ("probit", ProbitLink())]
        # FD backend only (for zero allocations)
        marginal_effects_mu!(g, Gβ, de_fd, β, link, 2)  # Warmup
        b_link = @benchmark marginal_effects_mu!($g, $Gβ, $de_fd, $β, $link, 3) samples=200
        push!(results, ("mu_marginal_effects_with_grad_$(link_name)", "fd", minimum(b_link.memory), minimum(b_link.times)))
        @test results[end, :min_memory_bytes] == 0
    end

    # === Variance computation utilities ===

    # delta_method_se (always 0 bytes - pure math)
    Σ_test = [1.0 0.1; 0.1 1.0]
    gβ_test = [0.5, -0.3]
    delta_method_se(gβ_test, Σ_test)  # Warmup
    b_delta = @benchmark delta_method_se($gβ_test, $Σ_test) samples=400
    push!(results, ("delta_method_se", "math", minimum(b_delta.memory), minimum(b_delta.times)))
    @test results[end, :min_memory_bytes] == 0


    # === Tight-loop tests to verify scaling behavior ===

    # η-scale with parameter gradients (FD backend)
    for _ in 1:10
        marginal_effects_eta!(g, Gβ, de_fd, β, 3)
    end
    b_loop_eta_grad = @benchmark begin
        for _ in 1:50_000
            marginal_effects_eta!($g, $Gβ, $de_fd, $β, 3)
        end
    end samples=20
    @test minimum(b_loop_eta_grad.memory) == 0

    # μ-scale with parameter gradients (FD backend, Logit)
    for _ in 1:10
        marginal_effects_mu!(g, Gβ, de_fd, β, LogitLink(), 3)
    end
    b_loop_mu_grad = @benchmark begin
        for _ in 1:50_000
            marginal_effects_mu!($g, $Gβ, $de_fd, $β, LogitLink(), 3)
        end
    end samples=20
    @test minimum(b_loop_mu_grad.memory) == 0

    # delta_method_se tight loop
    for _ in 1:10
        delta_method_se(gβ_test, Σ_test)
    end
    b_loop_delta = @benchmark begin
        for _ in 1:100_000
            delta_method_se($gβ_test, $Σ_test)
        end
    end samples=20
    @test minimum(b_loop_delta.memory) == 0

    # === Multi-row allocation validation (row-scaling check) ===

    # Test allocations across different rows to catch row-dependent allocations
    # This validates that categorical reference index optimization eliminates row-scaling allocations
    test_rows = [1, 50, 100, 150, 200, 250]

    @testset "Multi-row allocation validation" begin
        for test_row in test_rows
            # AD backend allocations
            b_ad = @benchmark derivative_modelrow!($Gβ, $de_ad, $test_row) samples=100 evals=1
            @test minimum(b_ad).memory == 0

            b_eta_ad = @benchmark marginal_effects_eta!($g, $Gβ, $de_ad, $β, $test_row) samples=100 evals=1
            @test minimum(b_eta_ad).memory == 0

            b_mu_ad = @benchmark marginal_effects_mu!($g, $Gβ, $de_ad, $β, LogitLink(), $test_row) samples=100 evals=1
            @test minimum(b_mu_ad).memory == 0

            # FD backend allocations
            b_fd = @benchmark derivative_modelrow!($Gβ, $de_fd, $test_row) samples=100 evals=1
            @test minimum(b_fd).memory == 0

            b_eta_fd = @benchmark marginal_effects_eta!($g, $Gβ, $de_fd, $β, $test_row) samples=100 evals=1
            @test minimum(b_eta_fd).memory == 0

            b_mu_fd = @benchmark marginal_effects_mu!($g, $Gβ, $de_fd, $β, LogitLink(), $test_row) samples=100 evals=1
            @test minimum(b_mu_fd).memory == 0
        end
    end

    # === Cross-validation: AD vs FD backends ===

    # Verify both backends produce consistent results (mathematical correctness)
    g_ad = Vector{Float64}(undef, length(vars))
    g_fd = Vector{Float64}(undef, length(vars))
    Gβ_ad = Matrix{Float64}(undef, length(compiled), length(vars))
    Gβ_fd = Matrix{Float64}(undef, length(compiled), length(vars))

    # Test multiple rows for robustness
    for test_row in [1, 50, 150, 299]
        # η-scale comparison
        marginal_effects_eta!(g_ad, Gβ_ad, de_ad, β, test_row)
        marginal_effects_eta!(g_fd, Gβ_fd, de_fd, β, test_row)

        @test g_ad ≈ g_fd rtol=1e-6 atol=1e-8
        @test Gβ_ad ≈ Gβ_fd rtol=1e-6 atol=1e-8

        # μ-scale comparison (Logit link)
        marginal_effects_mu!(g_ad, Gβ_ad, de_ad, β, LogitLink(), test_row)
        marginal_effects_mu!(g_fd, Gβ_fd, de_fd, β, LogitLink(), test_row)

        @test g_ad ≈ g_fd rtol=1e-6 atol=1e-8
        @test Gβ_ad ≈ Gβ_fd rtol=1e-6 atol=1e-8
    end
end
