using Test, Random
using DataFrames, Tables, GLM
using FormulaCompiler

@testset "FD log profile regression" begin
    # Synthetic data with strictly positive x to keep log in-domain
    Random.seed!(123)
    n = 500
    df = DataFrame(
        x = abs.(rand(n)) .+ 10.0,
        z = randn(n),
    )
    df.y = 0.5 .* log.(df.x) .+ 0.3 .* df.z .+ 0.1 .* randn(n)

    # Fit simple log model and compile
    model = lm(@formula(y ~ log(x)), df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)

    # Build derivative evaluator for x
    de = build_derivative_evaluator(compiled, data; vars=[:x])

    # Target row = 1 (regression covers prior row-1 initialization bug)
    row = 1
    xval = df.x[row]

    # 1) Single-column FD Jacobian should be finite and match analytic [0, 1/x]
    Jk = Vector{Float64}(undef, length(compiled))
    @test_nowarn fd_jacobian_column!(Jk, de, row, :x)
    @test Jk[1] ≈ 0.0 atol=1e-12
    @test Jk[2] ≈ 1.0 / xval rtol=1e-8

    # 2) Full FD Jacobian column equals analytic derivative
    J = Matrix{Float64}(undef, length(compiled), length(de.vars))
    @test_nowarn derivative_modelrow_fd!(J, de, row)
    @test J[1, 1] ≈ 0.0 atol=1e-12
    @test J[2, 1] ≈ 1.0 / xval rtol=1e-8

    # 3) η-profile β-gradient equals the single Jacobian column
    β = coef(model)
    gβ = Vector{Float64}(undef, length(compiled))
    @test_nowarn me_eta_grad_beta!(gβ, de, β, row, :x)
    @test gβ ≈ view(J, :, 1) rtol=1e-12 atol=1e-12

    # 4) μ-profile β-gradient with IdentityLink reduces to same Jacobian column
    gβ2 = Vector{Float64}(undef, length(compiled))
    @test_nowarn me_mu_grad_beta!(gβ2, de, β, row, :x; link=GLM.IdentityLink())
    @test gβ2 ≈ view(J, :, 1) rtol=1e-12 atol=1e-12

    # 5) AD marginal effects on η should match β1/x
    g = Vector{Float64}(undef, length(de.vars))
    @test_nowarn marginal_effects_eta!(g, de, β, row; backend=:ad)
    @test g[1] ≈ β[2] / xval rtol=1e-8
end

