using Documenter, FormulaCompiler

makedocs(
    sitename = "FormulaCompiler.jl",
    authors = "Eric Martin Feltham",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://emfeltham.github.io/FormulaCompiler.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Basic Usage" => "guide/basic_usage.md",
            "Advanced Features" => "guide/advanced_features.md",
            "Scenario Analysis" => "guide/scenarios.md",
            "Performance Tips" => "guide/performance.md",
        ],
        "Ecosystem Integration" => [
            "GLM.jl" => "integration/glm.md",
            "MixedModels.jl" => "integration/mixed_models.md",
            "StandardizedPredictors.jl" => "integration/standardized_predictors.md",
        ],
        "Mathematical Foundation" => "mathematical_foundation.md",
        "Architecture" => "architecture.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ],
    modules = [FormulaCompiler],
    warnonly = [:missing_docs, :docs_block]
)

deploydocs(
    repo = "github.com/emfeltham/FormulaCompiler.jl.git",
    devbranch = "main"
)