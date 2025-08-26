using Test
using FormulaCompiler
using DataFrames, Tables, GLM, CategoricalArrays
using BenchmarkTools

@testset "Derivative allocation checks (BenchmarkTools)" begin
    # Data and model similar to test_derivatives.jl
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

    # Warmup
    derivative_modelrow!(J, de, 2)
    derivative_modelrow_fd!(J_fd, compiled, data, 2; vars=vars)
    marginal_effects_eta_grad!(gη, de, β, 2)
    marginal_effects_mu!(gμ, de, β, 2; link=LogitLink())

    # FD Jacobian: expect 0 allocations
    b_fd = @benchmark derivative_modelrow_fd!($J_fd, $compiled, $data, 3; vars=$vars) samples=400
    @test minimum(b_fd.memory) == 0

    # ForwardDiff Jacobian: allow small FD-internal allocs (env dependent)
    b_ad = @benchmark derivative_modelrow!($J, $de, 3) samples=400
    @test minimum(b_ad.memory) <= 144

    # η-gradient path: fast scalar AD (allow small cap until config is hoisted)
    b_grad = @benchmark marginal_effects_eta_grad!($gη, $de, $β, 3) samples=400
    @test minimum(b_grad.memory) <= 192

    # μ marginal effects: follows η path + link scaling; cap conservatively
    b_mu = @benchmark marginal_effects_mu!($gμ, $de, $β, 3; link=LogitLink()) samples=400
    @test minimum(b_mu.memory) <= 256
end

