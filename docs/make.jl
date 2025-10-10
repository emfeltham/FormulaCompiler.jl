using Documenter, FormulaCompiler
using Glob

# Attempt to regenerate Mermaid diagrams from .mmd sources if the CLI is available.
function _regen_mermaid_assets()
    mmdc = Sys.which("mmdc")
    if isnothing(mmdc)
        @info "Mermaid CLI (mmdc) not found; skipping diagram regeneration"
        return
    end
    assets_dir = joinpath(@__DIR__, "src", "assets")
    for mmd in Glob.glob("*.mmd", assets_dir)
        svg = replace(mmd, ".mmd" => ".svg")
        try
            run(`$(mmdc) -i $(mmd) -o $(svg) -b transparent`)
            @info "Regenerated diagram" mmd svg
        catch e
            @warn "Failed to regenerate diagram; continuing" mmd exception=e
        end
    end
end

_regen_mermaid_assets()

makedocs(
    sitename = "FormulaCompiler.jl",
    authors = "Eric Martin Feltham",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://emfeltham.github.io/FormulaCompiler.jl",
        assets = String[
            "assets/mermaid.js",
        ],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Basic Usage" => "guide/basic_usage.md",
            "Advanced Features" => "guide/advanced_features.md",
            "Categorical Mixtures" => "guide/categorical_mixtures.md",
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
        "Metaprogramming" => "metaprogramming.md",
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
