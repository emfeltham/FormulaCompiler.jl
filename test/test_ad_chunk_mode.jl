# test_ad_chunk_mode.jl
# Regression tests for AD Jacobians in ForwardDiff chunk mode.
#
# With more than 12 variables, ForwardDiff splits the dual seeds into chunks and
# `fc_jacobian!` evaluates the formula once per chunk. Before 1.4.3 this path
# called ForwardDiff's internal `seed!(duals, x)`, which ForwardDiff 1.4.5 removed,
# so any AD evaluator with more than 12 variables threw a MethodError. Earlier
# tests only used vector mode (<= 12 variables) and did not catch it.
# The data mixes Float64 and Int columns: the AD counterfactuals must stay
# homogeneous (Float64 storage), or the evaluator dispatches dynamically and allocates.

using Test, Random, LinearAlgebra
using DataFrames, Tables, GLM
using ForwardDiff
using FormulaCompiler

# y ~ v1 + ... + vk + v1&v2 + v3&vk + log(v4): analytic Jacobian with respect to v1..vk.
# StatsModels orders columns by degree: intercept, v1..vk, log(v4), v1 & v2, v3 & vk.
function analytic_jacobian(df, row, k)
    v(i) = df[row, Symbol("v$i")]
    J = zeros(1 + k + 3, k)
    for i in 1:k
        J[1 + i, i] = 1.0
    end
    J[k + 2, 4] = 1 / v(4)                   # log(v4)
    J[k + 3, 1] = v(2); J[k + 3, 2] = v(1)   # v1 & v2
    J[k + 4, 3] = v(k); J[k + 4, k] = v(3)   # v3 & vk
    return J
end

# μ = η³ through FC's ForwardDiff fallback for links without analytic derivatives
struct CubeLinkAD <: GLM.Link end
GLM.linkinv(::CubeLinkAD, η::Real) = η^3

function ad_allocations(J, de, row)
    derivative_modelrow!(J, de, row)
    return @allocated derivative_modelrow!(J, de, row)
end

@testset "AD Jacobian in ForwardDiff chunk mode" begin
    Random.seed!(08540)
    n = 50
    # 12: vector mode (boundary); 13, 14: two chunks; 23: uneven last chunk; 25: three chunks
    for k in (12, 13, 14, 23, 25)
        @testset "$k variables (chunk size $(FormulaCompiler._ad_chunksize(k)))" begin
            df = DataFrame([Symbol("v$i") => randn(n) for i in 1:k])
            df.v4 = rand(n) .+ 0.5
            df.v5 = rand(1:9, n)   # Int column: mixed Int/Float64 inputs must stay allocation-free
            df.y = randn(n)
            mains = join(("v$i" for i in 1:k), " + ")
            f = eval(Meta.parse("@formula(y ~ $mains + v1 & v2 + v3 & v$k + log(v4))"))
            model = lm(f, df)
            data = Tables.columntable(df)
            compiled = compile_formula(model, data)
            vars = [Symbol("v$i") for i in 1:k]
            de = derivativeevaluator(:ad, compiled, data, vars)
            J = Matrix{Float64}(undef, length(compiled), k)

            for row in (1, 7, n)
                derivative_modelrow!(J, de, row)
                @test J ≈ analytic_jacobian(df, row, k) atol = 1e-12
            end

            # Repeated calls on different rows must not leak seeds between evaluations
            derivative_modelrow!(J, de, 3)
            derivative_modelrow!(J, de, 9)
            @test J ≈ analytic_jacobian(df, 9, k) atol = 1e-12

            @test ad_allocations(J, de, 2) == 0
        end
    end
end

# FC 1.5.0 (M2): the AD path no longer uses ForwardDiff internals (`JacobianConfig` fields,
# `partials(T, d, i)`, `pickchunksize`). FC owns its seeds, dual buffer and chunk-size rule;
# these tests check that the results are bitwise identical to ForwardDiff's own Jacobian
# with the same chunk size and tag, that the rule matches ForwardDiff's, that the path stays
# type-stable and allocation-free, and that FC's tag coexists with nested ForwardDiff use.
@testset "FC-owned AD configuration (public ForwardDiff API only)" begin
    FC = FormulaCompiler

    @testset "chunk size rule matches ForwardDiff's default" begin
        if isdefined(ForwardDiff, :pickchunksize)   # internal; only used here as an oracle
            for k in 0:200
                @test FC._ad_chunksize(k) == ForwardDiff.pickchunksize(k, 12)
            end
        end
        @test FC._ad_chunksize(12) == 12
        @test FC._ad_chunksize(13) == 7
        @test FC._ad_chunksize(25) == 9
    end

    @testset "seeds are the unit partials" begin
        seeds = FC._ad_seeds(ForwardDiff.Partials{4,Float64})
        @test length(seeds) == 4
        for i in 1:4, j in 1:4
            @test seeds[i][j] === (i == j ? 1.0 : 0.0)
        end
        @test FC._ad_seeds(ForwardDiff.Partials{0,Float64}) === ()
    end

    Random.seed!(20261016)
    n = 30
    for k in (1, 3, 12, 13, 25, 37)
        @testset "$k variables: identical to ForwardDiff.jacobian" begin
            df = DataFrame([Symbol("v$i") => randn(n) for i in 1:k])
            df.v1 = rand(n) .+ 0.5
            df.y = randn(n)
            mains = join(("v$i" for i in 1:k), " + ")
            f = eval(Meta.parse("@formula(y ~ $mains + log(v1) + v1 & v$k + exp(v$k / 4))"))
            model = lm(f, df)
            data = Tables.columntable(df)
            compiled = compile_formula(model, data)
            vars = [Symbol("v$i") for i in 1:k]
            de = derivativeevaluator(:ad, compiled, data, vars)
            N = FC._ad_chunksize(k)
            @test de.ctx.cfg isa FC.FCJacobianConfig{FC.FC_AD_TAG,Float64,N}
            J = Matrix{Float64}(undef, length(compiled), k)
            for row in (1, 11, n)
                derivative_modelrow!(J, de, row)          # also sets the evaluator's row
                x = Float64[data[v][row] for v in vars]
                cfg = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{N}(), FC.FC_AD_TAG())
                Jfd = ForwardDiff.jacobian(z -> copy(de.ctx.g(z)), x, cfg, Val{false}())
                @test size(Jfd) == size(J)
                @test all(J[i] === Jfd[i] for i in eachindex(J))   # bitwise
            end
            @test (@inferred derivative_modelrow!(J, de, 2)) === J
            @test ad_allocations(J, de, 4) == 0
        end
    end

    @testset "nested ForwardDiff use alongside FC's tag" begin
        df = DataFrame(x = randn(n), z = rand(n) .+ 1.0, y = randn(n))
        model = lm(@formula(y ~ x * z + log(z)), df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        de = derivativeevaluator(:ad, compiled, data, [:x, :z])
        J = Matrix{Float64}(undef, length(compiled), 2)
        β = coef(model)

        # Outer ForwardDiff derivative over β, FC's AD Jacobian inside: d/dt (β t)'J[:, 1] = β'J[:, 1]
        h(t) = dot(β .* t, derivative_modelrow!(J, de, 5)[:, 1])
        @test ForwardDiff.derivative(h, 1.7) ≈ dot(β, derivative_modelrow(de, 5)[:, 1])

        # FC's AD fallback for a user-defined link (itself ForwardDiff) differentiated by an
        # outer ForwardDiff call, interleaved with FC's own AD evaluations
        derivative_modelrow!(J, de, 3)
        @test ForwardDiff.derivative(e -> FC._dmu_deta(CubeLinkAD(), e), 0.7) ≈ 6 * 0.7
        derivative_modelrow!(J, de, 4)
        @test J ≈ derivative_modelrow(derivativeevaluator(:fd, compiled, data, [:x, :z]), 4) rtol = 1e-6

        # An outer gradient with its own tag through FC's link derivative, then FC AD again
        g = ForwardDiff.gradient(v -> FC._dmu_deta(LogitLink(), v[1]) * v[2], [0.2, 3.0])
        @test g[2] ≈ FC._dmu_deta(LogitLink(), 0.2)
        @test derivative_modelrow!(J, de, 6) === J
    end
end
