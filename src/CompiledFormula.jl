# CompiledFormula.jl

###############################################################################
# UPDATED COMPILE_FORMULA WITH @GENERATED INTEGRATION
###############################################################################

# Global cache for @generated functions
const FORMULA_CACHE = Dict{UInt64, Tuple{Vector{String}, Vector{Symbol}, Int}}()

"""
    CompiledFormula{H}

Enhanced compiled formula that stores the root evaluator for ecosystem extensibility.

# Fields
- `formula_val::Val{H}`: Hash-based identifier for @generated function dispatch
- `output_width::Int`: Width of output vector
- `column_names::Vector{Symbol}`: Variable names from formula
- `root_evaluator::AbstractEvaluator`: Root of the evaluator tree (NEW!)

# Benefits of Storing Root Evaluator
- **Derivative compilation**: Enables analytical derivatives (Margins.jl)
- **Symbolic integration**: Convert to symbolic form (Symbolics.jl)
- **Custom backends**: Compile to GPU, C, WebAssembly, etc.
- **Formula introspection**: Analyze computational complexity
- **Meta-programming**: Build tools that manipulate formulas

# Example
```julia
model = lm(@formula(y ~ x + log(z) + x*group), df)
compiled = compile_formula(model)

# Direct access to evaluator tree
evaluator = compiled.root_evaluator
println("Formula complexity: ", count_nodes(evaluator))

# Build derivatives (Margins.jl)
derivative_compiled = compile_derivative_formula(compiled, :x)

# Custom analysis
variables = get_variable_dependencies(compiled)
```
"""
struct CompiledFormula{H}
    formula_val::Val{H}
    output_width::Int
    column_names::Vector{Symbol}
    root_evaluator::AbstractEvaluator  # NEW: Store the evaluator tree
end

Base.length(cf::CompiledFormula) = cf.output_width
variables(cf::CompiledFormula) = cf.column_names

# Call interface - delegate to @generated function
function (cf::CompiledFormula)(row_vec::AbstractVector{Float64}, data, row_idx::Int)
    _modelrow_generated!(row_vec, cf.formula_val, data, row_idx)
    return row_vec
end

"""
    Updated compile_formula function that stores root evaluator in CompiledFormula.
    
    Key change: Instead of discarding the root_evaluator after code generation,
    we store it in the CompiledFormula for future use.
"""
function compile_formula(model; verbose = false)
    
    if verbose
        println("=== Compiling Formula with Root Evaluator Storage ===")
    end
    
    rhs = fixed_effects_form(model).rhs
    
    # Step 1: Build recursive evaluator tree (unchanged)
    root_evaluator = compile_term(rhs)
    total_width = output_width(root_evaluator)
    column_names = extract_all_columns(rhs)
    
    if verbose
        println("Built evaluator tree: width=$total_width, columns=$column_names")
        println("Root evaluator type: $(typeof(root_evaluator))")
    end
    
    # Step 2: Generate code strings from evaluator tree (unchanged)
    instructions = generate_code_from_evaluator(root_evaluator)
    
    if verbose
        println("Generated $(length(instructions)) instructions:")
        for (i, instr) in enumerate(instructions[1:min(5, length(instructions))])
            println("  $i: $instr")
        end
        if length(instructions) > 5
            println("  ... ($(length(instructions) - 5) more)")
        end
    end
    
    # Step 3: Cache for @generated function (unchanged)
    formula_hash = hash(string(rhs))
    FORMULA_CACHE[formula_hash] = (instructions, column_names, total_width)
    
    if verbose
        println("Cached with hash: $formula_hash")
    end
    
    # Step 4: Return CompiledFormula WITH root evaluator
    return CompiledFormula(Val(formula_hash), total_width, column_names, root_evaluator)
end

"""
@generated function for zero-allocation model row evaluation.
FIXED: More specific signature to avoid ambiguity.
"""
@generated function _modelrow_generated!(
    row_vec::AbstractVector{Float64}, 
    ::Val{formula_hash}, 
    data, 
    row_idx::Int;
    verbose = false
) where formula_hash
    
    # Retrieve instructions from cache
    if !haskey(FORMULA_CACHE, formula_hash)
        error("Formula hash $formula_hash not found in cache")
    end
    
    instructions, column_names, output_width = FORMULA_CACHE[formula_hash]
    
    if verbose == true
        println("@generated: Compiling for hash $formula_hash with $(length(instructions)) instructions")
    end
    
    # Convert instruction strings to expressions
    try
        code_exprs = [Meta.parse(line) for line in instructions]
        
        return quote
            @inbounds begin
                $(code_exprs...)
            end
            return row_vec
        end
        
    catch e
        error("Failed to parse instructions for hash $formula_hash: $e")
    end
end
