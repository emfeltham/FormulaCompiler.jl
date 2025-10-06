# Case 3 — MixedModels Fixed Effects (LMM)

using Dates
using DataFrames, Tables, CategoricalArrays, Random
using MixedModels
using FormulaCompiler

let
    println("Case: MixedModels Fixed Effects (LMM)")
    println("Julia:", VERSION, " | Threads:", Threads.nthreads())
    println()

    Random.seed!(44)
    n = 10_000
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        treatment = rand(Bool, n),
        group = categorical(rand(1:200, n)),
    )

    # Random intercept model
    f = @formula(y ~ x + treatment + (1 | group))
    m = fit(MixedModel, f, df)

    data = Tables.columntable(df)
    compiled = compile_formula(m, data)  # fixed effects only
    row = Vector{Float64}(undef, length(compiled))
    compiled(row, data, 1)

    # Derivatives and marginal effects for fixed effects
    vars = continuous_variables(compiled, data)
    de_fd = derivativeevaluator(:fd, compiled, data, vars)
    β = collect(fixef(m))
    i = 50
    g = Vector{Float64}(undef, length(vars))
    marginal_effects_eta!(g, de_fd, β, i)
    println("η-scale ME (FD) @row ", i, ": ", g)

    # Parameter-gradient for single-row ME and SE
    gβ_row = zeros(Float64, length(β))
    me_eta_grad_beta!(gβ_row, de_fd, i)
    se_row = delta_method_se(gβ_row, Matrix{Float64}(vcov(m)))
    println("Single-row ME SE (eta): ", se_row)

    # Artifact
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    out = joinpath("results", "case_mixedmodels_fixed_" * ts * ".md")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "# Case 3 — MixedModels Fixed Effects (LMM)")
        println(io, "Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io, "\nVars: ", vars)
        println(io, "\nη-scale ME (FD) @row ", i, ": ", g)
        println(io, "\nSingle-row ME SE (eta): ", se_row)
    end
    println("Wrote Case 3 results to ", out)
end

