using Test, Random
using DataFrames, Tables, GLM
using FormulaCompiler

@testset "Derivative domain edge cases" begin
    Random.seed!(08540)

    # 1) log1p(x) near -1 (but in-domain)
    @testset "log1p near -1" begin
        df = DataFrame(x = fill(-0.9, 5), y = randn(5))
        model = lm(@formula(y ~ log1p(x)), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        de = build_derivative_evaluator(compiled, data; vars=[:x])
        row = 1
        xval = df.x[row]
        Jk = Vector{Float64}(undef, length(compiled))
        @test_nowarn fd_jacobian_column!(Jk, de, row, :x)
        @test Jk[1] ≈ 0.0 atol=1e-12
        @test Jk[2] ≈ 1.0 / (1.0 + xval) rtol=1e-8
    end

    # 2) sqrt(x) near 0 (small positive)
    @testset "sqrt near 0" begin
        df = DataFrame(x = [1e-3, 1e-2, 1e-1], y = randn(3))
        model = lm(@formula(y ~ sqrt(x)), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        de = build_derivative_evaluator(compiled, data; vars=[:x])
        row = 1
        xval = df.x[row]
        # Prefer AD validation near the boundary for accuracy
        J_ad = Matrix{Float64}(undef, length(compiled), length(de.vars))
        @test_nowarn derivative_modelrow!(J_ad, de, row)
        @test J_ad[1, 1] ≈ 0.0 atol=1e-12
        @test J_ad[2, 1] ≈ 0.5 / sqrt(xval) rtol=1e-10
        # FD should at least run without DomainError; accuracy is known to be poorer here
        Jk = Vector{Float64}(undef, length(compiled))
        @test_nowarn fd_jacobian_column!(Jk, de, row, :x)
    end

    # 3) inverse via power: x^-1
    @testset "inverse x^-1" begin
        df = DataFrame(x = fill(0.25, 5), y = randn(5))
        model = lm(@formula(y ~ x^-1), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        de = build_derivative_evaluator(compiled, data; vars=[:x])
        row = 1
        xval = df.x[row]
        Jk = Vector{Float64}(undef, length(compiled))
        @test_nowarn fd_jacobian_column!(Jk, de, row, :x)
        # d/dx x^-1 = -x^-2
        @test Jk[2] ≈ -(xval^-2) rtol=1e-8
    end

    # 4) non-integer power: x^0.7
    @testset "power x^0.7" begin
        df = DataFrame(x = fill(2.3, 5), y = randn(5))
        model = lm(@formula(y ~ x^0.7), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        de = build_derivative_evaluator(compiled, data; vars=[:x])
        row = 1
        xval = df.x[row]
        Jk = Vector{Float64}(undef, length(compiled))
        @test_nowarn fd_jacobian_column!(Jk, de, row, :x)
        # d/dx x^a = a*x^(a-1)
        @test Jk[2] ≈ 0.7 * xval^(-0.3) rtol=1e-8
    end

    # 5) shifted domain: log(x - c) with x just above c
    @testset "shifted log(x - c)" begin
        c = 10.0
        testx = 10.5
        df = DataFrame(x = fill(testx, 3), y = randn(3))
        model = lm(@formula(y ~ log(x - 10.0)), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        de = build_derivative_evaluator(compiled, data; vars=[:x])
        row = 1
        xval = df.x[row]
        Jk = Vector{Float64}(undef, length(compiled))
        @test_nowarn fd_jacobian_column!(Jk, de, row, :x)
        @test Jk[2] ≈ 1.0 / (xval - c) rtol=1e-8
    end

    # 6) integer column with log
    @testset "integer x with log(x)" begin
        df = DataFrame(x = Int.(fill(12, 4)), y = randn(4))
        model = lm(@formula(y ~ log(x)), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        de = build_derivative_evaluator(compiled, data; vars=[:x])
        row = 1
        xval = float(df.x[row])
        Jk = Vector{Float64}(undef, length(compiled))
        @test_nowarn fd_jacobian_column!(Jk, de, row, :x)
        @test Jk[2] ≈ 1.0 / xval rtol=1e-8
    end

    # 7) interaction with log: y ~ z * log(x)
    @testset "interaction z*log(x)" begin
        df = DataFrame(x = fill(10.5, 5), z = fill(2.0, 5), y = randn(5))
        model = lm(@formula(y ~ z * log(x)), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        de = build_derivative_evaluator(compiled, data; vars=[:x, :z])
        row = 1
        @test length(compiled) == 4  # intercept, z, log(x), z*log(x)
        xval = df.x[row]; zval = df.z[row]
        # FD Jacobian and AD Jacobian should match closely
        J_fd = Matrix{Float64}(undef, length(compiled), length(de.vars))
        J_ad = Matrix{Float64}(undef, length(compiled), length(de.vars))
        @test_nowarn derivative_modelrow_fd!(J_fd, de, row)
        @test_nowarn derivative_modelrow!(J_ad, de, row)
        @test J_fd ≈ J_ad rtol=1e-8 atol=1e-10

        # Analytic expectations:
        # ∂/∂x [1, z, log(x), z*log(x)] = [0, 0, 1/x, z*(1/x)]
        # ∂/∂z [1, z, log(x), z*log(x)] = [0, 1, 0, log(x)]
        expected_dx = [0.0, 0.0, 1.0/xval, zval*(1.0/xval)]
        expected_dz = [0.0, 1.0, 0.0, log(xval)]
        @test J_fd[:, 1] ≈ expected_dx rtol=1e-8 atol=1e-10
        @test J_fd[:, 2] ≈ expected_dz rtol=1e-8 atol=1e-10

        # β-gradient for η equals single-column
        β = coef(model)
        gβ_x = similar(J_fd, size(J_fd,1))[:,1]; resize!(gβ_x, size(J_fd,1))
        gβ_buf = Vector{Float64}(undef, length(compiled))
        @test_nowarn me_eta_grad_beta!(gβ_buf, de, β, row, :x)
        @test gβ_buf ≈ view(J_fd, :, 1) rtol=1e-12 atol=1e-12
        @test_nowarn me_eta_grad_beta!(gβ_buf, de, β, row, :z)
        @test gβ_buf ≈ view(J_fd, :, 2) rtol=1e-12 atol=1e-12
    end
end
