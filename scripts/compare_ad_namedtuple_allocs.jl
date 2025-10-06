#!/usr/bin/env julia
# julia --project="test" scripts/compare_ad_namedtuple_allocs.jl > scripts/compare_ad_namedtuple_allocs.txt 2>&1

using BenchmarkTools
using DataFrames
using Tables
using GLM
using FormulaCompiler
using ForwardDiff

include(joinpath(@__DIR__, "..", "test", "support", "generate_large_synthetic_data.jl"))

function build_model(n; seed=1)
    df = generate_synthetic_dataset(n; seed)
    data = Tables.columntable(df)
    fx = @formula(response ~
        socio4 +
        (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
        !socio4 & dists_a_inv +
        (schoolyears_p + wealth_d1_4_p + man_p + age_p + religion_c_p +
        same_building + population +
        hhi_religion + hhi_indigenous +
        coffee_cultivation + market + relation) & (1 + socio4 + are_related_dists_a_inv) +
        (degree_a_mean + degree_h +
        age_a_mean + age_h * age_h_nb_1_socio +
        schoolyears_a_mean + schoolyears_h * schoolyears_h_nb_1_socio +
        man_x * man_x_mixed_nb_1 +
        wealth_d1_4_a_mean + wealth_d1_4_h * wealth_d1_4_h_nb_1_socio +
        isindigenous_x * isindigenous_homop_nb_1 + religion_c_x * religion_homop_nb_1) & (1 + socio4 + are_related_dists_a_inv) +
        religion_c_x & hhi_religion +
        isindigenous_x & hhi_indigenous)
    model = fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())
    compiled = compile_formula(model, data)
    return compiled, data
end

function benchmark_derivative_modelrow(n; seed=1)
    compiled, data = build_model(n; seed)
    vars = [:age_h, :dists_p_inv, :are_related_dists_a_inv, :schoolyears_h]
    de = derivativeevaluator(:ad, compiled, data, vars)
    J = Matrix{Float64}(undef, length(compiled), length(vars))

    # warmup
    for _ in 1:5
        derivative_modelrow!(J, de, 1)
    end

    ctx = de.ctx
    core = de.core
    core.row = 1
    for i in eachindex(de.vars)
        base_col = getproperty(de.base_data, de.vars[i])
        ctx.input_vec[i] = Float64(base_col[1])
    end

    b_dm = @benchmark derivative_modelrow!($J, $de, 1)
    b_fd = @benchmark ForwardDiff.jacobian!($J, $(ctx.g), $(ctx.input_vec), $(ctx.cfg))

    return (; n,
            columns = length(keys(data)),
            derivative_min_bytes = minimum(b_dm).memory,
            derivative_trial = b_dm,
            fd_min_bytes = minimum(b_fd).memory,
            fd_trial = b_fd)
end

results_small = benchmark_derivative_modelrow(30; seed=123)
results_large = benchmark_derivative_modelrow(200; seed=123)

open(joinpath(@__DIR__, "compare_ad_namedtuple_allocs.txt"), "w") do io
    println(io, "== Small dataset ==")
    println(io, "rows: ", results_small.n)
    println(io, "columns: ", results_small.columns)
    println(io, "alloc derivative_modelrow!: ", results_small.derivative_min_bytes)
    println(io, "alloc ForwardDiff.jacobian!: ", results_small.fd_min_bytes)
    show(io, MIME("text/plain"), results_small.derivative_trial)
    println(io)
    println(io)

    println(io, "== Large dataset ==")
    println(io, "rows: ", results_large.n)
    println(io, "columns: ", results_large.columns)
    println(io, "alloc derivative_modelrow!: ", results_large.derivative_min_bytes)
    println(io, "alloc ForwardDiff.jacobian!: ", results_large.fd_min_bytes)
    show(io, MIME("text/plain"), results_large.derivative_trial)
    println(io)
end

println("Wrote results to scripts/compare_ad_namedtuple_allocs.txt")
