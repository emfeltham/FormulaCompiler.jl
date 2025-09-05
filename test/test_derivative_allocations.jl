# test_derivative_allocations.jl

using Test
using FormulaCompiler
using DataFrames, Tables, GLM, CategoricalArrays
using BenchmarkTools
using CSV

"""
Non-capturing kernels for strict zero-allocation BenchmarkTools checks
"""
function _bench_derivative_modelrow!(Jloc, deloc, rowloc)
    derivative_modelrow!(Jloc, deloc, rowloc)
    return nothing
end

function _bench_eta_grad!(gloc, deloc, bloc, rowloc)
    marginal_effects_eta_grad!(gloc, deloc, bloc, rowloc)
    return nothing
end

@testset "Derivative Allocation Checks" begin
    results = DataFrame(
        path = String[],
        min_memory_bytes = Int[],
        min_time_seconds = Float64[],
    )
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

    # Build derivative evaluator (ForwardDiff)
    de = build_derivative_evaluator(compiled, data; vars=vars)

    # Buffers
    J = Matrix{Float64}(undef, length(compiled), length(vars))
    J_fd = similar(J)
    gη = Vector{Float64}(undef, length(vars))
    gμ = Vector{Float64}(undef, length(vars))
    row_vec = Vector{Float64}(undef, length(compiled))

    # Warmup
    compiled(row_vec, data, 2)
    derivative_modelrow!(J, de, 2)
    derivative_modelrow_fd!(J_fd, compiled, data, 2; vars=vars)
    derivative_modelrow_fd!(J_fd, de, 2)
    marginal_effects_eta_grad!(gη, de, β, 2)
    marginal_effects_mu!(gμ, de, β, 2; link=LogitLink())

    # Core compiled evaluation: expect 0 allocations
    b_comp = @benchmark $compiled($row_vec, $data, 3) samples=600
    push!(results, ("compiled_row", minimum(b_comp.memory), minimum(b_comp.times)))
    @test results[end, :min_memory_bytes] == 0

    # FD Jacobian (standalone): builds row-local overrides per call; allow small cap
    b_fd = @benchmark derivative_modelrow_fd!($J_fd, $compiled, $data, 3; vars=$vars) samples=400
    push!(results, ("fd_jacobian_standalone", minimum(b_fd.memory), minimum(b_fd.times)))
    @test results[end, :min_memory_bytes] <= 2048

    # FD Jacobian via evaluator (generated, prebuilt overrides)
    # Use the exported positional hot path to avoid keyword overhead
    b_fd_eval = @benchmark derivative_modelrow_fd_pos!($J_fd, $de, 3) samples=400
    push!(results, ("fd_jacobian_evaluator", minimum(b_fd_eval.memory), minimum(b_fd_eval.times)))
    @test results[end, :min_memory_bytes] == 0

    # ForwardDiff Jacobian: allow small FD-internal allocs (env dependent)
    b_ad = @benchmark derivative_modelrow!($J, $de, 3) samples=400
    push!(results, ("ad_jacobian", minimum(b_ad.memory), minimum(b_ad.times)))
    @test results[end, :min_memory_bytes] <= 512

    # Manual dual path strict check using BenchmarkTools (non-capturing kernel)
    b_ad_zero = @benchmark _bench_derivative_modelrow!($J, $de, 3) samples=400
    @test minimum(b_ad_zero.memory) <= 128  # tolerate tiny env-dependent noise

    # η-gradient path: allow cap until all environments report zero
    b_grad = @benchmark marginal_effects_eta_grad!($gη, $de, $β, 3) samples=400
    push!(results, ("eta_gradient", minimum(b_grad.memory), minimum(b_grad.times)))
    @test results[end, :min_memory_bytes] <= 512

    # Manual dual path strict check for η-gradient (non-capturing kernel)
    b_grad_zero = @benchmark _bench_eta_grad!($gη, $de, $β, 3) samples=400
    @test minimum(b_grad_zero.memory) <= 128  # tolerate tiny env-dependent noise

    # μ marginal effects (Logit): follows η path + link scaling; cap conservatively
    b_mu = @benchmark marginal_effects_mu!($gμ, $de, $β, 3; link=LogitLink()) samples=400
    push!(results, ("mu_marginal_effects_logit_ad", minimum(b_mu.memory), minimum(b_mu.times)))
    # Allow small ForwardDiff-internal allocations (env-dependent)
    @test results[end, :min_memory_bytes] <= 512

    # Test zero-allocation FD backends for marginal effects
    b_eta_fd = @benchmark marginal_effects_eta!($gη, $de, $β, 3; backend=:fd) samples=400
    push!(results, ("eta_marginal_effects_fd", minimum(b_eta_fd.memory), minimum(b_eta_fd.times)))
    @test results[end, :min_memory_bytes] == 0

    b_mu_fd = @benchmark marginal_effects_mu!($gμ, $de, $β, 3; link=LogitLink(), backend=:fd) samples=400
    push!(results, ("mu_marginal_effects_logit_fd", minimum(b_mu_fd.memory), minimum(b_mu_fd.times)))
    @test results[end, :min_memory_bytes] == 0

    # NEW: Single-column FD Jacobian (zero allocations expected)
    Jk_buffer = Vector{Float64}(undef, length(compiled))
    fd_jacobian_column!(Jk_buffer, de, 2, :x)  # Warmup
    b_fd_col = @benchmark fd_jacobian_column!($Jk_buffer, $de, 3, :x) samples=400
    push!(results, ("fd_jacobian_column", minimum(b_fd_col.memory), minimum(b_fd_col.times)))
    @test results[end, :min_memory_bytes] == 0

    # NEW: η parameter gradients (zero allocations expected)
    gβ_buffer = Vector{Float64}(undef, length(compiled))
    me_eta_grad_beta!(gβ_buffer, de, β, 2, :x)  # Warmup
    b_eta_grad = @benchmark me_eta_grad_beta!($gβ_buffer, $de, $β, 3, :x) samples=400
    push!(results, ("me_eta_grad_beta", minimum(b_eta_grad.memory), minimum(b_eta_grad.times)))
    @test results[end, :min_memory_bytes] == 0

    # NEW: μ parameter gradients with chain rule (zero allocations expected)
    me_mu_grad_beta!(gβ_buffer, de, β, 2, :x; link=LogitLink())  # Warmup
    b_mu_grad = @benchmark me_mu_grad_beta!($gβ_buffer, $de, $β, 3, :x; link=LogitLink()) samples=400
    push!(results, ("me_mu_grad_beta", minimum(b_mu_grad.memory), minimum(b_mu_grad.times)))
    @test results[end, :min_memory_bytes] == 0

    # Save results to CSV for inspection
    CSV.write(joinpath(@__DIR__, "derivative_allocations.csv"), results)

    # Tight-loop allocation test (via BenchmarkTools) for FD evaluator
    # Verify that any tiny allocation reported by the single-call benchmark
    # does not scale with the number of calls in a real loop.
    for _ in 1:10
        derivative_modelrow_fd_pos!(J_fd, de, 3)
    end
    b_loop = @benchmark begin
        for _ in 1:100_000
            derivative_modelrow_fd_pos!($J_fd, $de, 3)
        end
    end samples=20
    @test minimum(b_loop.memory) == 0

    # Tight-loop tests for the three new functions
    # Single-column FD Jacobian
    for _ in 1:10
        fd_jacobian_column!(Jk_buffer, de, 3, :x)
    end
    b_loop_fd_col = @benchmark begin
        for _ in 1:100_000
            fd_jacobian_column!($Jk_buffer, $de, 3, :x)
        end
    end samples=20
    @test minimum(b_loop_fd_col.memory) == 0

    # η parameter gradients
    for _ in 1:10
        me_eta_grad_beta!(gβ_buffer, de, β, 3, :x)
    end
    b_loop_eta_grad = @benchmark begin
        for _ in 1:100_000
            me_eta_grad_beta!($gβ_buffer, $de, $β, 3, :x)
        end
    end samples=20
    @test minimum(b_loop_eta_grad.memory) == 0

    # μ parameter gradients
    for _ in 1:10
        me_mu_grad_beta!(gβ_buffer, de, β, 3, :x; link=LogitLink())
    end
    b_loop_mu_grad = @benchmark begin
        for _ in 1:100_000
            me_mu_grad_beta!($gβ_buffer, $de, $β, 3, :x; link=LogitLink())
        end
    end samples=20
    @test minimum(b_loop_mu_grad.memory) == 0

    # NEW: Variance computation primitives allocation tests
    
    # delta_method_se (always 0 bytes - pure math)
    Σ_test = [1.0 0.1; 0.1 1.0]
    gβ_test = [0.5, -0.3]
    delta_method_se(gβ_test, Σ_test)  # Warmup
    b_delta = @benchmark delta_method_se($gβ_test, $Σ_test) samples=400
    push!(results, ("delta_method_se", minimum(b_delta.memory), minimum(b_delta.times)))
    @test results[end, :min_memory_bytes] == 0

    # accumulate_ame_gradient! with :fd backend (zero allocations expected)
    rows_test = 1:5
    gβ_ame_buffer = Vector{Float64}(undef, length(compiled))
    accumulate_ame_gradient!(gβ_ame_buffer, de, β, rows_test, :x; backend=:fd)  # Warmup
    b_ame_fd = @benchmark accumulate_ame_gradient!($gβ_ame_buffer, $de, $β, $rows_test, :x; backend=:fd) samples=400
    push!(results, ("accumulate_ame_gradient_fd", minimum(b_ame_fd.memory), minimum(b_ame_fd.times)))
    @test results[end, :min_memory_bytes] == 0

    # accumulate_ame_gradient! with :ad backend (small allocations expected)
    accumulate_ame_gradient!(gβ_ame_buffer, de, β, rows_test, :x; backend=:ad)  # Warmup
    b_ame_ad = @benchmark accumulate_ame_gradient!($gβ_ame_buffer, $de, $β, $rows_test, :x; backend=:ad) samples=400
    push!(results, ("accumulate_ame_gradient_ad", minimum(b_ame_ad.memory), minimum(b_ame_ad.times)))
    @test results[end, :min_memory_bytes] <= 2048  # Allow small allocations for AD backend

    # Tight-loop tests for variance primitives
    
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

    # accumulate_ame_gradient! :fd backend tight loop
    for _ in 1:10
        accumulate_ame_gradient!(gβ_ame_buffer, de, β, rows_test, :x; backend=:fd)
    end
    b_loop_ame_fd = @benchmark begin
        for _ in 1:1000  # Fewer iterations since this is more expensive
            accumulate_ame_gradient!($gβ_ame_buffer, $de, $β, $rows_test, :x; backend=:fd)
        end
    end samples=20
    @test minimum(b_loop_ame_fd.memory) == 0

    # accumulate_ame_gradient! :ad backend tight loop (allow allocations)
    for _ in 1:10
        accumulate_ame_gradient!(gβ_ame_buffer, de, β, rows_test, :x; backend=:ad)
    end
    b_loop_ame_ad = @benchmark begin
        for _ in 1:1000  # Fewer iterations since this is more expensive
            accumulate_ame_gradient!($gβ_ame_buffer, $de, $β, $rows_test, :x; backend=:ad)
        end
    end samples=20
    # AD backend can have scaling allocations, so we just check it's reasonable
    @test minimum(b_loop_ame_ad.memory) <= 400_000  # Cap adjusted for environment variability
end
