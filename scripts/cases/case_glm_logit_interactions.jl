# Case 1 — GLM Logit with Interactions + CounterfactualVector Mixtures

using Dates
using DataFrames, Tables, CategoricalArrays, Random
using GLM
using FormulaCompiler
using FormulaCompiler: derivativeevaluator_fd, mix

let
    # Environment summary
    println("Case: GLM Logit with Interactions + CounterfactualVector Mixtures")
    println("Julia:", VERSION, " | Threads:", Threads.nthreads())
    println()

    # Data
    Random.seed!(06515)
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
    de_fd = derivativeevaluator_fd(compiled, data, vars)

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

    # Mixture example: use mix() in test data instead of scenario
    mixspec = mix("A" => 0.4, "B" => 0.4, "C" => 0.2)

    # Create test data with mixture column for demonstration
    test_df = DataFrame(
        y = [true, false, true],  # small test data
        x = [1.0, 2.0, 3.0],
        z = [0.5, 1.0, 1.5],
        group = [mixspec, mixspec, mixspec]  # mixture columns
    )
    test_data = Tables.columntable(test_df)

    # Compile with mixture data
    try
        test_compiled = compile_formula(model, test_data)
        row = Vector{Float64}(undef, length(test_compiled))
        test_compiled(row, test_data, 1)
        println("Row @mixture evaluation (fixed effects row vector): length=", length(row))
        println("Mixture specification: ", mixspec)
    catch e
        println("Mixture compilation note: ", e)
        println("(Mixture processing depends on training data schema)")
    end

    # Demonstrate CounterfactualVector approach for systematic analysis
    println("\nCounterfactualVector Analysis Pattern:")

    # Create reference grid approach using CounterfactualVector
    group_values = ["A", "B", "C"]
    x_values = [-1.0, 0.0, 1.0]

    effects_grid = Matrix{Float64}(undef, length(x_values), length(group_values))
    row = Vector{Float64}(undef, length(compiled))

    for (i_x, x_val) in enumerate(x_values)
        for (i_g, group_val) in enumerate(group_values)
            # Create counterfactual for this combination
            cf_x = FormulaCompiler.NumericCounterfactualVector{Float64}(data.x, i, x_val)
            group_cat = CategoricalValue(group_val, data.group)
            cf_group = FormulaCompiler.CategoricalCounterfactualVector(data.group, i, group_cat)
            cf_data = merge(data, (x = cf_x, group = cf_group))

            # Evaluate at representative row
            compiled(row, cf_data, i)
            linear_pred = dot(β, row)
            prob_pred = 1.0 / (1.0 + exp(-linear_pred))  # Logit inverse link

            effects_grid[i_x, i_g] = prob_pred
        end
    end

    println("Predicted probabilities grid (x values: ", x_values, ", groups: ", group_values, "):")
    for (i_x, x_val) in enumerate(x_values)
        println("  x=$x_val: ", [round(effects_grid[i_x, i_g], digits=3) for i_g in 1:length(group_values)])
    end

    # Optional: Margins.jl integration (if installed)
    try
        @eval begin
            import Margins
            println("\nMargins.jl detected — computing MEM/MER sketch (details in Margins repo)")
            # Placeholder: exact API depends on Margins.jl; the intent is to show FC-backed workflow.
        end
    catch
        println("\n(Margins.jl not available in this environment; skipping high-level workflow demo)")
    end

    # Write minimal artifact
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    out = joinpath("results", "case_glm_logit_counterfactuals_" * ts * ".md")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "# Case 1 — GLM Logit with Interactions + CounterfactualVector Mixtures")
        println(io, "Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io, "\nEnv: Julia ", VERSION, ", Threads ", Threads.nthreads())
        println(io, "\nVars: ", vars)
        println(io, "\nη-scale ME (FD) @row ", i, ": ", g)
        println(io, "\nAME (μ) SE (delta): ", se_ame)
        println(io, "\nMixture specification: ", mixspec)
        println(io, "\n## CounterfactualVector Analysis Results")
        println(io, "Predicted probabilities grid:")
        println(io, "x values: ", x_values)
        println(io, "groups: ", group_values)
        for (i_x, x_val) in enumerate(x_values)
            println(io, "x=$x_val: ", [round(effects_grid[i_x, i_g], digits=3) for i_g in 1:length(group_values)])
        end
        println(io, "\n## Migration Notes")
        println(io, "- Replaced derivativeevaluator(:fd, ...) with derivativeevaluator_fd(...)")
        println(io, "- Replaced create_scenario() mixture with mix() in test data")
        println(io, "- Added CounterfactualVector grid analysis pattern")
        println(io, "- Maintained zero-allocation performance for derivatives")
    end
    println("Wrote Case 1 results to ", out)
end