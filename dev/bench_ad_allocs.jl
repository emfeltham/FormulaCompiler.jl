#!/usr/bin/env julia
## Assuming julia is launched with --project=.

using BenchmarkTools
using FormulaCompiler
using StatsModels
using GLM

function build_fixture(n::Int)
    x = randn(n)
    z = randn(n)
    y = 1 .+ 2 .* x .+ 0.5 .* z .+ 0.3 .* (x .* z) .+ randn(n) .* 0.01
    data = (y=y, x=x, z=z)
    formula = @formula(y ~ 1 + x + z + x & z + log(abs(z) + 1))
    compiled = compile_formula(formula, data)
    return compiled, data
end

function bench_ad_zero_alloc(n::Int=10_000; row::Int=1)
    compiled, data = build_fixture(n)
    vars = [:x, :z]
    de_ad = derivativeevaluator(:ad, compiled, data, vars)

    # Preallocate buffers
    J = Matrix{Float64}(undef, length(compiled), length(vars))
    g = Vector{Float64}(undef, length(vars))
    β = randn(length(compiled))

    # Warmup
    derivative_modelrow!(J, de_ad, row)
    marginal_effects_eta_grad!(g, de_ad, β, row)

    println("— AD Jacobian allocations (expect 0 after warmup) —")
    jac_bench = @benchmark derivative_modelrow!($J, $de_ad, $row)
    show(stdout, MIME("text/plain"), jac_bench); println()
    println("min bytes:", minimum(jac_bench).memory)

    println("— AD η-gradient allocations (β::Vector{Float64}, expect 0) —")
    eta_bench_vec = @benchmark marginal_effects_eta_grad!($g, $de_ad, $β, $row)
    show(stdout, MIME("text/plain"), eta_bench_vec); println()
    println("min bytes:", minimum(eta_bench_vec).memory)

    # Test non-Vector beta path (uses internal beta_buf)
    β_view = view(β, :)
    println("— AD η-gradient allocations (β::SubArray, expect 0) —")
    eta_bench_view = @benchmark marginal_effects_eta_grad!($g, $de_ad, $β_view, $row)
    show(stdout, MIME("text/plain"), eta_bench_view); println()
    println("min bytes:", minimum(eta_bench_view).memory)

    return (; jac_bench, eta_bench_vec, eta_bench_view)
end

if abspath(PROGRAM_FILE) == @__FILE__
    bench_ad_zero_alloc()
end
