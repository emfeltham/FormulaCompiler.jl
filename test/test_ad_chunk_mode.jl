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

using Test, Random
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

function ad_allocations(J, de, row)
    derivative_modelrow!(J, de, row)
    return @allocated derivative_modelrow!(J, de, row)
end

@testset "AD Jacobian in ForwardDiff chunk mode" begin
    Random.seed!(08540)
    n = 50
    # 12: vector mode (boundary); 13, 14: two chunks; 23: uneven last chunk; 25: three chunks
    for k in (12, 13, 14, 23, 25)
        @testset "$k variables (chunk size $(ForwardDiff.pickchunksize(k)))" begin
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
