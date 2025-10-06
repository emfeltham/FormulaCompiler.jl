# Case 2 — Linear Model + CounterfactualVector Loop Patterns (formerly scenarios)

using Dates
using DataFrames, Tables, CategoricalArrays, Random
using GLM
using FormulaCompiler
using FormulaCompiler: NumericCounterfactualVector, CategoricalCounterfactualVector, derivativeevaluator_fd

let
    println("Case: Linear Model + CounterfactualVector Loop Patterns")
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

    # CounterfactualVector: policy shift in x (e.g., +0.5) and group forced to "B"
    cf_x = NumericCounterfactualVector{Float64}(data.x, 1, 0.5)
    group_b = CategoricalValue("B", data.group)
    cf_group = CategoricalCounterfactualVector(data.group, 1, group_b)
    cf_data = merge(data, (x = cf_x, group = cf_group))

    # Evaluate a few rows baseline vs counterfactual using loop pattern
    idxs = [10, 100, 1000]
    effects = Float64[]
    for i in idxs
        # Baseline evaluation
        compiled(row, data, i)
        y0 = dot(coef(model), row)

        # Counterfactual evaluation - update which row gets the counterfactual
        FormulaCompiler.update_counterfactual_row!(cf_x, i)
        FormulaCompiler.update_counterfactual_row!(cf_group, i)
        compiled(row, cf_data, i)
        y1 = dot(coef(model), row)

        push!(effects, y1 - y0)
    end
    println("Counterfactual effects at rows ", idxs, ": ", effects)

    # Marginal effects and SE via delta method (η scale)
    vars = continuous_variables(compiled, data)
    de_fd = derivativeevaluator_fd(compiled, data, vars)
    β = collect(coef(model))
    i = 25
    gη = Vector{Float64}(undef, length(vars))
    marginal_effects_eta!(gη, de_fd, β, i)

    gβ = zeros(Float64, length(β))
    rows = 1:2000
    # Note: Need to specify which variable for AME gradient
    var_for_ame = vars[1]  # Use first continuous variable
    accumulate_ame_gradient!(gβ, de_fd, β, rows, var_for_ame)
    gβ ./= length(rows)
    se_ame = delta_method_se(gβ, Matrix{Float64}(vcov(model)))
    println("AME (η) SE (delta method) for ", var_for_ame, " over ", length(rows), " rows: ", se_ame)

    # Demonstrate efficient population analysis loop pattern
    println("\nPopulation Analysis Pattern:")
    population_effects = Float64[]
    sample_rows = [100, 500, 1000, 5000, 10000]  # Sample different rows

    for target_row in sample_rows
        # Apply counterfactual to this specific row
        FormulaCompiler.update_counterfactual_row!(cf_x, target_row)
        FormulaCompiler.update_counterfactual_row!(cf_group, target_row)

        # Evaluate effect at this row
        compiled(row, data, target_row)
        y0 = dot(coef(model), row)
        compiled(row, cf_data, target_row)
        y1 = dot(coef(model), row)

        push!(population_effects, y1 - y0)
    end

    avg_population_effect = sum(population_effects) / length(population_effects)
    println("Average population effect across sampled rows: ", avg_population_effect)
    println("Population effects by row: ", collect(zip(sample_rows, population_effects)))

    # Artifact
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    out = joinpath("results", "case_lm_counterfactuals_" * ts * ".md")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "# Case 2 — Linear Model + CounterfactualVector Loop Patterns")
        println(io, "Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io, "\n## Individual Row Analysis")
        println(io, "Rows sampled: ", idxs)
        println(io, "Counterfactual effects: ", effects)
        println(io, "\n## Population Analysis")
        println(io, "Sample rows: ", sample_rows)
        println(io, "Population effects: ", population_effects)
        println(io, "Average population effect: ", avg_population_effect)
        println(io, "\n## Derivatives")
        println(io, "Vars: ", vars)
        println(io, "AME (η) SE (delta): ", se_ame)
        println(io, "\n## Migration Notes")
        println(io, "- Replaced create_scenario() with CounterfactualVector")
        println(io, "- Updated derivativeevaluator(:fd, ...) to derivativeevaluator_fd(...)")
        println(io, "- Added population analysis loop pattern")
        println(io, "- Maintained zero-allocation performance")
    end
    println("Wrote Case 2 results to ", out)
end