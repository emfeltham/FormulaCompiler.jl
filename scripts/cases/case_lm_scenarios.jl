# Case 2 — Linear Model + Policy Scenarios (Counterfactuals)

using Dates
using DataFrames, Tables, CategoricalArrays, Random
using GLM
using FormulaCompiler

let
    println("Case: Linear Model + Policy Scenarios")
    println("Julia:", VERSION, " | Threads:", Threads.nthreads())
    println()

    # Data
    Random.seed!(43)
    n = 50_000
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n)),
    )

    # Gaussian LM with interaction
    f = @formula(y ~ x + z + x * group)
    model = lm(f, df)

    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    row = Vector{Float64}(undef, length(compiled))

    # Scenario: policy shift in x (e.g., +0.5) and group forced to "B"
    scen = create_scenario("policy", data; x = 0.5, group = "B")

    # Evaluate a few rows baseline vs scenario
    idxs = [10, 100, 1000]
    effects = Float64[]
    for i in idxs
        compiled(row, data, i)
        y0 = dot(coef(model), row)
        compiled(row, scen.data, i)
        y1 = dot(coef(model), row)
        push!(effects, y1 - y0)
    end
    println("Scenario effects at rows ", idxs, ": ", effects)

    # Marginal effects and SE via delta method (η scale)
    vars = continuous_variables(compiled, data)
    de = build_derivative_evaluator(compiled, data; vars=vars)
    β = collect(coef(model))
    i = 25
    gη = Vector{Float64}(undef, length(vars))
    marginal_effects_eta!(gη, de, β, i; backend=:fd)

    gβ = zeros(Float64, length(β))
    rows = 1:2000
    accumulate_ame_gradient!(gβ, de, β, rows; backend=:fd)
    gβ ./= length(rows)
    se_ame = delta_method_se(gβ, Matrix{Float64}(vcov(model)))
    println("AME (η) SE (delta method) over ", length(rows), " rows: ", se_ame)

    # Artifact
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    out = joinpath("results", "case_lm_scenarios_" * ts * ".md")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "# Case 2 — Linear Model + Policy Scenarios")
        println(io, "Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io, "\nRows sampled: ", idxs)
        println(io, "\nScenario effects: ", effects)
        println(io, "\nVars: ", vars)
        println(io, "\nAME (η) SE (delta): ", se_ame)
    end
    println("Wrote Case 2 results to ", out)
end

