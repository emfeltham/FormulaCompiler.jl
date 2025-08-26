# Script to check docstrings for all exported functions
# Run with: julia --project=. check_docs.jl

println("\n" * "="^60)
println("DOCSTRING STATUS FOR EXPORTED FUNCTIONS")
println("="^60)

# List of exported items from src/FormulaCompiler.jl
exported_functions = [
    # Core compilation
    ("compile_formula", "Function"),
    
    # Model row evaluation
    ("modelrow", "Function"),
    ("modelrow!", "Function"),
    ("ModelRowEvaluator", "Type"),
    
    # Override and scenario system
    ("OverrideVector", "Type"),
    ("DataScenario", "Type"),
    ("create_scenario", "Function"),
    ("create_scenario_grid", "Function"),
    ("create_override_data", "Function"),
    ("create_override_vector", "Function"),
    ("create_categorical_override", "Function"),
    
    # Utility
    ("not", "Function")
]

println("\n📋 Checking $(length(exported_functions)) exported items...\n")

# Track statistics
global has_docs = 0
global missing_docs = 0
global needs_improvement = String[]

for (name, kind) in exported_functions
    print("• $name ($kind): ")
    
    # Check if it's actually exported
    if occursin("export.*$name", read("src/FormulaCompiler.jl", String))
        print("✅ Exported")
    else
        print("⚠️  Not exported")
    end
    
    # Search for docstring pattern
    files = String[]
    for (root, dirs, filenames) in walkdir("src")
        for file in filenames
            if endswith(file, ".jl")
                push!(files, joinpath(root, file))
            end
        end
    end
    
    found_docstring = false
    for file in files
        content = read(file, String)
        
        # Look for docstring patterns before the function/type definition
        patterns = [
            r"\"\"\"\n.*?"*name*r".*?\n.*?\"\"\""s,  # Triple quote docstring
            r"\"\"\".*?"*name*r".*?\"\"\""s,          # One-line triple quote
        ]
        
        for pattern in patterns
            if occursin(pattern, content)
                found_docstring = true
                
                # Check quality of docstring
                match_content = match(pattern, content)
                if match_content !== nothing
                    doc_text = match_content.match
                    
                    # Check for important elements
                    has_description = length(doc_text) > 50
                    has_arguments = occursin("Arguments", doc_text) || occursin("#", doc_text)
                    has_returns = occursin("Returns", doc_text) || occursin("return", lowercase(doc_text))
                    has_example = occursin("Example", doc_text) || occursin("```", doc_text)
                    
                    if kind == "Function" && (!has_arguments || !has_returns)
                        push!(needs_improvement, "$name (missing args/returns info)")
                    end
                end
                break
            end
        end
        
        if found_docstring
            break
        end
    end
    
    if found_docstring
        println(" | ✅ Has docstring")
        global has_docs += 1
    else
        println(" | ❌ Missing docstring")
        global missing_docs += 1
    end
end

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("✅ With docstrings:    $has_docs / $(length(exported_functions))")
println("❌ Missing docstrings: $missing_docs / $(length(exported_functions))")

if length(needs_improvement) > 0
    println("\n⚠️  Docstrings that could be improved:")
    for item in needs_improvement
        println("   - $item")
    end
end

println("\n" * "="^60)
println("RECOMMENDATIONS")
println("="^60)

if missing_docs > 0
    println("\n🔴 Critical: Add docstrings for:")
    for (name, kind) in exported_functions
        found = false
        for file in readdir("src", join=true, recursive=true)
            if endswith(file, ".jl") && occursin(r"\"\"\""*name, read(file, String))
                found = true
                break
            end
        end
        if !found
            println("   - $name")
        end
    end
end

println("\n📝 Docstring Template:")
println("""
\"\"\"
    function_name(arg1, arg2; kwarg=default)

Brief description of what the function does.

# Arguments
- `arg1`: Description of first argument
- `arg2`: Description of second argument
- `kwarg`: Description of keyword argument (default: value)

# Returns
- Description of return value

# Example
```julia
result = function_name(value1, value2)
```
\"\"\"
""")