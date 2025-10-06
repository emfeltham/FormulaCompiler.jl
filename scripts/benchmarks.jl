# Benchmark scaffold for FormulaCompiler.jl
# Usage examples:
#   julia --project=. scripts/benchmarks.jl            # run default suite
#   julia --project=. scripts/benchmarks.jl core deriv margins se

using BenchmarkTools
using Dates
using FormulaCompiler
using GLM
using MixedModels
using Tables
using DataFrames
using CategoricalArrays
using ForwardDiff
using Random
using LinearAlgebra
import TOML
using FormulaCompiler: NumericCounterfactualVector, CategoricalCounterfactualVector, derivativeevaluator_fd, derivativeevaluator_ad, mix

# -----------------------------
# Utilities
# -----------------------------

struct BenchResult
    name::String
    median_ns::Float64
    minimum_ns::Float64
    min_alloc_bytes::Int
end

function summarize_trial(name::String, t::BenchmarkTools.Trial)
    return BenchResult(
        name,
        BenchmarkTools.median(t).time,
        BenchmarkTools.minimum(t).time,
        BenchmarkTools.minimum(t).memory,
    )
end

function print_result(io::IO, r::BenchResult)
    println(io, rpad(r.name, 36), " | median: ", round(r.median_ns; digits=1), " ns",
            " | min: ", round(r.minimum_ns; digits=1), " ns",
            " | min mem: ", r.min_alloc_bytes, " B")
end

function env_summary(io::IO=stdout)
    println(io, "Environment Summary")
    println(io, "- Julia: ", VERSION)
    println(io, "- Threads: ", Threads.nthreads())
    try
        println(io, "- CPU: ", get(ENV, "CPU_NAME", Sys.CPU_NAME))
    catch
    end
    println(io, "- OS: ", Sys.KERNEL)
    # Safe package version lookup via Project.toml parsing
    _pkgver(mod) = try
        dir = Base.pkgdir(mod)
        toml = joinpath(dir, "Project.toml")
        if isfile(toml)
            try
                parsed = TOML.parsefile(toml)
                v = get(parsed, "version", nothing)
                isnothing(v) ? "unknown" : string(v)
            catch
                "unknown"
            end
        else
            "unknown"
        end
    catch
        "unknown"
    end
    println(io, "- Packages: FormulaCompiler ", _pkgver(FormulaCompiler),
            ", GLM ", _pkgver(GLM),
            ", MixedModels ", _pkgver(MixedModels),
            ", ForwardDiff ", _pkgver(ForwardDiff))
    println(io)
end

# ---------------------------------
# Emitters (Markdown / CSV)
# ---------------------------------

function emit_markdown(path::AbstractString, results::Vector{BenchResult}; tag::AbstractString="")
    mkpath(dirname(path))
    open(path, "w") do io
        ts = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")
        println(io, "# Benchmark Results", isempty(tag) ? "" : " — " * tag)
        println(io)
        println(io, "Generated: ", ts)
        println(io)
        # Environment summary
        println(io, "## Environment")
        println(io, "- Julia: ", VERSION)
        println(io, "- Threads: ", Threads.nthreads())
        try
            println(io, "- CPU: ", get(ENV, "CPU_NAME", Sys.CPU_NAME))
        catch
        end
        println(io, "- OS: ", Sys.KERNEL)
        # Safe package version lookup via Project.toml parsing
        _pkgver(mod) = try
            dir = Base.pkgdir(mod)
            toml = joinpath(dir, "Project.toml")
            if isfile(toml)
                try
                    parsed = TOML.parsefile(toml)
                    v = get(parsed, "version", nothing)
                    isnothing(v) ? "unknown" : string(v)
                catch
                    "unknown"
                end
            else
                "unknown"
            end
        catch
            "unknown"
        end
        println(io, "- Packages: FormulaCompiler ", _pkgver(FormulaCompiler),
                    ", GLM ", _pkgver(GLM),
                    ", MixedModels ", _pkgver(MixedModels),
                    ", ForwardDiff ", _pkgver(ForwardDiff))
        println(io)
        # Results
        println(io, "## Results")
        println(io, "| Name | Median (ns) | Min (ns) | Min Mem (B) |")
        println(io, "|------|-------------|----------|-------------|")
        for r in results
            println(io, "| ", r.name, " | ", round(r.median_ns; digits=1), " | ", round(r.minimum_ns; digits=1), " | ", r.min_alloc_bytes, " |")
        end
    end
end

function emit_csv(path::AbstractString, results::Vector{BenchResult}; tag::AbstractString="")
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "name,median_ns,minimum_ns,min_alloc_bytes,tag")
        for r in results
            println(io, string(r.name, ",", round(r.median_ns; digits=1), ",", round(r.minimum_ns; digits=1), ",", r.min_alloc_bytes, ",", tag))
        end
    end
end

# -----------------------------
# Data and models
# -----------------------------

function make_data(n::Int=20_000; seed::Int=42)
    Random.seed!(seed)
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n)),
        b = rand(Bool, n),
    )
    return df
end

function fit_glm(df::DataFrame)
    f = @formula(y ~ x * group + log1p(z))
    return lm(f, df)
end

function fit_lmm(df::DataFrame)
    # Small random intercept model for speed
    f = @formula(y ~ x + (1 | group))
    return fit(MixedModel, f, df)
end

function build_compiled(model, df::DataFrame)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    row = Vector{Float64}(undef, length(compiled))
    return (; data, compiled, row)
end

# -----------------------------
# Benchmarks
# -----------------------------

function bench_core(; n=50_000)
    df = make_data(n)
    model = fit_glm(df)
    ctx = build_compiled(model, df)
    compiled, data, row = ctx.compiled, ctx.data, ctx.row
    compiled(row, data, 1) # warmup
    t = @benchmark $compiled($row, $data, 25)
    return summarize_trial("core: compiled(row,data,i)", t)
end

function bench_alloc_vs_inplace(; n=10_000)
    df = make_data(n)
    model = fit_glm(df)
    data = Tables.columntable(df)
    # In-place path
    compiled = compile_formula(model, data)
    row = Vector{Float64}(undef, length(compiled))
    compiled(row, data, 1) # warmup
    t_inp = @benchmark $compiled($row, $data, 10)
    # Allocating wrapper
    t_alloc = @benchmark modelrow($model, $data, 10)
    return (
        summarize_trial("in-place: compiled(row,data,i)", t_inp),
        summarize_trial("allocating: modelrow(model,data,i)", t_alloc),
    )
end

function bench_counterfactual_overhead(; n=50_000)
    df = make_data(n)
    model = fit_glm(df)
    ctx = build_compiled(model, df)
    compiled, data, row = ctx.compiled, ctx.data, ctx.row
    lvl = first(levels(df.group))
    # Create counterfactual vectors
    cf_x = NumericCounterfactualVector{Float64}(data.x, 1, 2.0)
    group_val = CategoricalValue(string(lvl), data.group)
    cf_group = CategoricalCounterfactualVector(data.group, 1, group_val)
    cf_data = merge(data, (x = cf_x, group = cf_group))
    compiled(row, data, 1) # warmup
    compiled(row, cf_data, 1)
    t_base = @benchmark $compiled($row, $data, 25)
    t_cf = @benchmark $compiled($row, $cf_data, 25)
    return (
        summarize_trial("counterfactual: baseline", t_base),
        summarize_trial("counterfactual: CounterfactualVector data", t_cf),
    )
end

function bench_derivatives(; n=20_000)
    df = make_data(n)
    model = fit_glm(df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    vars = continuous_variables(compiled, data)
    de_ad = derivativeevaluator_ad(compiled, data, vars)
    de_fd = derivativeevaluator_fd(compiled, data, vars)
    i = 25
    # AD full Jacobian
    J = Matrix{Float64}(undef, length(compiled), length(vars))
    derivative_modelrow!(J, de_ad, i) # warmup
    t_ad = @benchmark derivative_modelrow!($J, $de_ad, $i)
    # FD single column (first variable by index)
    col = Vector{Float64}(undef, length(compiled))
    var_idx = 1
    fd_jacobian_column!(col, de_fd, i, var_idx) # warmup
    t_fd = @benchmark fd_jacobian_column!($col, $de_fd, $i, $var_idx)
    return (
        summarize_trial("deriv: AD full J", t_ad),
        summarize_trial("deriv: FD single col", t_fd),
    )
end

function bench_marginal_effects(; n=20_000)
    df = make_data(n)
    model = fit_glm(df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    vars = continuous_variables(compiled, data)
    de_fd = derivativeevaluator_fd(compiled, data, vars)
    de_ad = derivativeevaluator_ad(compiled, data, vars)
    β = collect(coef(model))
    i = 25
    g = Vector{Float64}(undef, length(vars))
    # FD (η)
    marginal_effects_eta!(g, de_fd, β, i) # warmup
    t_eta_fd = @benchmark marginal_effects_eta!($g, $de_fd, $β, $i)
    # AD (η)
    marginal_effects_eta!(g, de_ad, β, i) # warmup
    t_eta_ad = @benchmark marginal_effects_eta!($g, $de_ad, $β, $i)
    # μ (Logit as example link)
    t_mu_fd = @benchmark marginal_effects_mu!($g, $de_fd, $β, $i, LogitLink())
    t_mu_ad = @benchmark marginal_effects_mu!($g, $de_ad, $β, $i, LogitLink())
    return (
        summarize_trial("ME η: FD", t_eta_fd),
        summarize_trial("ME η: AD", t_eta_ad),
        summarize_trial("ME μ (Logit): FD", t_mu_fd),
        summarize_trial("ME μ (Logit): AD", t_mu_ad),
    )
end

function bench_delta_se(; n=5_000)
    df = make_data(n)
    model = fit_glm(df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    vars = continuous_variables(compiled, data)
    de_fd = derivativeevaluator_fd(compiled, data, vars)
    β = collect(coef(model))
    i = 25
    # Use η-scale gradient w.r.t β as example: gβ = J[:, var1]
    J = Matrix{Float64}(undef, length(compiled), length(vars))
    derivative_modelrow!(J, de_fd, i)
    gβ = view(J, :, 1)
    Σ = Matrix{Float64}(vcov(model))
    # Warm and bench
    _ = delta_method_se(gβ, Σ)
    t = @benchmark delta_method_se($gβ, $Σ)
    return summarize_trial("delta method SE", t)
end

function bench_mixedmodels(; n=10_000)
    df = make_data(n)
    m = fit_lmm(df)
    data = Tables.columntable(df)
    compiled = compile_formula(m, data)
    row = Vector{Float64}(undef, length(compiled))
    compiled(row, data, 1) # warmup
    t = @benchmark $compiled($row, $data, 25)
    return summarize_trial("mixedmodels: compiled(row,data,i)", t)
end

function bench_scaling(; n=30_000)
    df = make_data(n)
    # Simple vs more complex formula
    m_simple = lm(@formula(y ~ x + z + group), df)
    m_complex = lm(@formula(y ~ x * group + log1p(z) + x & z + x & log1p(z)), df)
    ctx_s = build_compiled(m_simple, df)
    ctx_c = build_compiled(m_complex, df)
    ctx_s.compiled(ctx_s.row, ctx_s.data, 1)
    ctx_c.compiled(ctx_c.row, ctx_c.data, 1)
    t_s = @benchmark $(ctx_s.compiled)($(ctx_s.row), $(ctx_s.data), 25)
    t_c = @benchmark $(ctx_c.compiled)($(ctx_c.row), $(ctx_c.data), 25)
    return (
        summarize_trial("scaling: simple", t_s),
        summarize_trial("scaling: complex", t_c),
    )
end

function bench_scale_n(; ns=(10_000, 100_000, 1_000_000))
    results = BenchResult[]
    for n in ns
        df = make_data(n)
        # Complex, interaction-heavy formula; fit model to apply schema (categoricals)
        formula = @formula(y ~ x * group + log1p(z) + x & z + x & log1p(z))
        model = lm(formula, df)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        row = Vector{Float64}(undef, length(compiled))
        i = min(n, 25)
        compiled(row, data, i) # warmup
        t = @benchmark $compiled($row, $data, $i)
        push!(results, summarize_trial("scale_n: n=$(n)", t))
    end
    return results
end

function bench_mixtures(; n=20_000)
    df = make_data(n)
    model = fit_glm(df)
    # Create test data with mixture column instead of scenario
    test_df = DataFrame(
        y = randn(3),
        x = randn(3),
        z = abs.(randn(3)) .+ 0.1,
        group = [mix("A" => 0.4, "B" => 0.4, "C" => 0.2) for _ in 1:3]
    )
    test_data = Tables.columntable(test_df)
    compiled = compile_formula(model, test_data)
    row = Vector{Float64}(undef, length(compiled))
    compiled(row, test_data, 1) # warmup
    t = @benchmark $compiled($row, $test_data, 1)
    return summarize_trial("mixture: compiled(row,mixture_data,i)", t)
end

# -----------------------------
# Runner
# -----------------------------

const ALL_BENCHES = (
    :core,
    :alloc,
    :scenario,
    :deriv,
    :margins,
    :se,
    :mixed,
    :scaling,
    :scale_n,
    :mixtures,
)

function run_selected(selected::Vector{Symbol}; fast::Bool=false)
    env_summary()
    results = BenchResult[]
    for b in selected
        if b == :core
            push!(results, fast ? bench_core(n=5_000) : bench_core())
        elseif b == :alloc
            r1, r2 = fast ? bench_alloc_vs_inplace(n=3_000) : bench_alloc_vs_inplace()
            append!(results, [r1, r2])
        elseif b == :scenario
            r1, r2 = fast ? bench_counterfactual_overhead(n=5_000) : bench_counterfactual_overhead()
            append!(results, [r1, r2])
        elseif b == :deriv
            r1, r2 = fast ? bench_derivatives(n=5_000) : bench_derivatives()
            append!(results, [r1, r2])
        elseif b == :margins
            r1, r2, r3, r4 = fast ? bench_marginal_effects(n=5_000) : bench_marginal_effects()
            append!(results, [r1, r2, r3, r4])
        elseif b == :se
            push!(results, fast ? bench_delta_se(n=2_000) : bench_delta_se())
        elseif b == :mixed
            push!(results, fast ? bench_mixedmodels(n=3_000) : bench_mixedmodels())
        elseif b == :scaling
            r1, r2 = fast ? bench_scaling(n=5_000) : bench_scaling()
            append!(results, [r1, r2])
        elseif b == :scale_n
            rs = bench_scale_n()
            append!(results, rs)
        elseif b == :mixtures
            push!(results, fast ? bench_mixtures(n=5_000) : bench_mixtures())
        else
            @warn "Unknown benchmark key" b
        end
    end
    println("Name                                | median | min | min mem")
    for r in results
        print_result(stdout, r)
    end
    return results
end

function parse_args()
    # Simple flag parser: supports --out=md|csv, --file=path, --tag=string, --fast, and bench keys
    out::Union{Nothing,Symbol} = nothing
    file::Union{Nothing,String} = nothing
    tag::String = ""
    fast::Bool = false
    bench_syms = Symbol[]
    for a in ARGS
        if startswith(a, "--out=")
            v = split(a, "=", limit=2)[2]
            if v in ("md", "markdown")
                out = :md
            elseif v in ("csv")
                out = :csv
            else
                @warn "Unknown --out value" v
            end
        elseif startswith(a, "--file=")
            file = split(a, "=", limit=2)[2]
        elseif startswith(a, "--tag=")
            tag = split(a, "=", limit=2)[2]
        elseif a == "--fast" || a == "--fast=true" || a == "--fast=1"
            fast = true
        elseif a == "--fast=false" || a == "--fast=0"
            fast = false
        else
            s = Symbol(a)
            if s ∉ ALL_BENCHES
                @warn "Ignoring unknown benchmark key" a
            else
                push!(bench_syms, s)
            end
        end
    end
    isempty(bench_syms) && (bench_syms = [:core, :alloc, :scenario, :deriv, :margins, :se])
    return (; out, file, tag, fast, bench_syms)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args()
    results = run_selected(args.bench_syms; fast=args.fast)
    if !isnothing(args.out)
        ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
        out_file = args.file
        if isnothing(out_file)
            mkpath("results")
            ext = args.out === :md ? ".md" : ".csv"
            tagpart = isempty(args.tag) ? "" : "_" * args.tag
            out_file = joinpath("results", "benchmarks_" * ts * tagpart * ext)
        end
        if args.out === :md
            emit_markdown(out_file, results; tag=args.tag)
            println("\nWrote Markdown results to ", out_file)
        elseif args.out === :csv
            emit_csv(out_file, results; tag=args.tag)
            println("\nWrote CSV results to ", out_file)
        end
    end
end
