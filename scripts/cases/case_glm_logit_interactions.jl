# Case 1 — GLM Logit with Interactions + Mixtures

using Dates
using DataFrames, Tables, CategoricalArrays, Random
using GLM
using FormulaCompiler

let
    # Environment summary
    println("Case: GLM Logit with Interactions + Mixtures")
    println("Julia:", VERSION, " | Threads:", Threads.nthreads())
    println()

    # Data
    Random.seed!(42)
    n = 20_000
    df = DataFrame(
        y = rand(Bool, n),
        x = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n)),
    )

    # Model with interactions
    f = @formula(y ~ x * group + log1p(z))
    model = glm(f, df, Binomial(), LogitLink())

    # Compile + derivative evaluator
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    vars = continuous_variables(compiled, data)
    de_fd = derivativevaluator(:fd, compiled, data, vars)

    # Representative row
    i = min(25, n)
    β = collect(coef(model))
    g = Vector{Float64}(undef, length(vars))

    # Marginal effects (η)
    marginal_effects_eta!(g, de_fd, β, i)
    println("η-scale ME (FD) @row ", i, ": ", g)

    # Marginal effects (μ) with Logit link
    marginal_effects_mu!(g, de_fd, β, i, LogitLink())
    println("μ-scale ME (FD) @row ", i, ": ", g)

    # AME gradient accumulation (μ via chain rule in evaluator), SE via delta method
    gβ = zeros(Float64, length(β))
    rows = 1:1000
    # Note: Need to specify which variable for AME gradient
    var_for_ame = vars[1]  # Use first continuous variable
    accumulate_ame_gradient!(gβ, de_fd, β, rows, var_for_ame)
    gβ ./= length(rows)
    se_ame = delta_method_se(gβ, Matrix{Float64}(vcov(model)))
    println("AME (μ) SE (delta method) for ", var_for_ame, " over first ", length(rows), " rows: ", se_ame)

    # Mixture example: profile effects for group
    mixspec = mix("A" => 0.4, "B" => 0.4, "C" => 0.2)
    scen = create_scenario("mixture_profile", data; group=mixspec)
    row = Vector{Float64}(undef, length(compiled))
    compiled(row, scen.data, i)
    println("Row @mixture scenario (fixed effects row vector): length=", length(row))

    # Optional: Margins.jl integration (if installed)
    try
        @eval begin
            import Margins
            println("Margins.jl detected — computing MEM/MER sketch (details in Margins repo)")
            # Placeholder: exact API depends on Margins.jl; the intent is to show FC-backed workflow.
        end
    catch
        println("(Margins.jl not available in this environment; skipping high-level workflow demo)")
    end

    # Write minimal artifact
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    out = joinpath("results", "case_glm_logit_interactions_" * ts * ".md")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "# Case 1 — GLM Logit with Interactions + Mixtures")
        println(io, "Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io, "\nEnv: Julia ", VERSION, ", Threads ", Threads.nthreads())
        println(io, "\nVars: ", vars)
        println(io, "\nη-scale ME (FD) @row ", i, ": ", g)
        println(io, "\nAME (μ) SE (delta): ", se_ame)
        println(io, "\nMixture: ", mixspec)
    end
    println("Wrote Case 1 results to ", out)
end

