# Test zero-allocation contract for batch marginal effects API
using Test, FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays, BenchmarkTools

@testset "Batch API Allocation Tests" begin
    # Setup test data
    n = 1000
    df = DataFrame(
        y = rand(Bool, n),
        x1 = randn(n),
        x2 = randn(n),
        x3 = randn(n),
        group = categorical(rand(["A", "B"], n))
    )

    model = glm(@formula(y ~ x1 + x2 + x3 + group), df, Binomial(), LogitLink())
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    vars = [:x1, :x2]
    de_fd = derivativeevaluator(:fd, compiled, data, vars)
    de_ad = derivativeevaluator(:ad, compiled, data, vars)
    β = coef(model)

    # Pre-allocate caller accumulators
    n_vars, n_params = length(vars), length(β)
    ame_values = Vector{Float64}(undef, n_vars)
    gradients = Matrix{Float64}(undef, n_vars, n_params)
    var_indices = Vector{Int}(undef, n_vars)
    rows = 1:100

    @testset "FD Backend Allocations" begin
        # Warmup to eliminate compilation allocations
        marginal_effects_batch!(
            ame_values, gradients, var_indices,
            de_fd, β, rows, vars;
            link=LogitLink(), backend=:fd, scale=:response
        )
        marginal_effects_batch!(
            ame_values, gradients, var_indices,
            de_fd, β, rows, vars;
            link=LogitLink(), backend=:fd, scale=:response
        )

        # Measure allocations after warmup using BenchmarkTools
        b_fd_resp = @benchmark begin
            FormulaCompiler.marginal_effects_batch!($ame_values, $gradients, $var_indices,
                                                   $de_fd, $β, $rows, $vars;
                                                   link=LogitLink(), backend=:fd, scale=:response)
        end samples=100 evals=1
        @test minimum(b_fd_resp.memory) == 0

        b_fd_lin = @benchmark begin
            FormulaCompiler.marginal_effects_batch!($ame_values, $gradients, $var_indices,
                                                   $de_fd, $β, $rows, $vars;
                                                   backend=:fd, scale=:linear)
        end samples=100 evals=1
        @test minimum(b_fd_lin.memory) == 0
    end

    @testset "AD Backend Allocations" begin
        # Warmup
        marginal_effects_batch!(
            ame_values, gradients, var_indices,
            de_ad, β, rows, vars;
            link=LogitLink(), backend=:ad, scale=:response
        )
        marginal_effects_batch!(
            ame_values, gradients, var_indices,
            de_ad, β, rows, vars;
            link=LogitLink(), backend=:ad, scale=:response
        )

        # Test allocations after warmup
        b_ad_resp = @benchmark begin
            FormulaCompiler.marginal_effects_batch!($ame_values, $gradients, $var_indices,
                                                   $de_ad, $β, $rows, $vars;
                                                   link=LogitLink(), backend=:ad, scale=:response)
        end samples=100 evals=1
        @test minimum(b_ad_resp.memory) == 0

        b_ad_lin = @benchmark begin
            FormulaCompiler.marginal_effects_batch!($ame_values, $gradients, $var_indices,
                                                   $de_ad, $β, $rows, $vars;
                                                   backend=:ad, scale=:linear)
        end samples=100 evals=1
        @test minimum(b_ad_lin.memory) == 0
    end

    @testset "Correctness Verification" begin
        # Compare results between old per-row approach and new batch API

        # Manual computation (reference)
        ame_manual = zeros(Float64, n_vars)
        grad_manual = zeros(Float64, n_vars, n_params)

        for row in rows
            # Marginal effects
            g_buf = Vector{Float64}(undef, length(de_fd.vars))
            marginal_effects_mu!(g_buf, de_fd, β, row, LogitLink())

            # Find indices
            for (i, var) in enumerate(vars)
                idx = findfirst(==(var), de_fd.vars)
                ame_manual[i] += g_buf[idx]
            end

            # Gradients
            for (i, var) in enumerate(vars)
                grad_buf = Vector{Float64}(undef, n_params)
                me_mu_grad_beta!(grad_buf, de_fd, β, row, var, LogitLink())
                grad_manual[i, :] .+= grad_buf
            end
        end

        # Average
        ame_manual /= length(rows)
        grad_manual /= length(rows)

        # Batch computation
        fill!(ame_values, 0.0)
        fill!(gradients, 0.0)
        marginal_effects_batch!(
            ame_values, gradients, var_indices,
            de_fd, β, rows, vars;
            link=LogitLink(), backend=:fd, scale=:response
        )

        # Compare results
        @test ame_values ≈ ame_manual rtol=1e-12
        @test gradients ≈ grad_manual rtol=1e-12
    end
end
