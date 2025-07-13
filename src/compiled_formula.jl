# compiled_formula.jl - New clean struct-based interface

"""
    CompiledFormula{H}

A compiled formula representation that provides zero-allocation model matrix row evaluation.

# Fields
- `formula_val::Val{H}`: Type-stable formula hash for @generated dispatch
- `output_width::Int`: Number of columns in the model matrix
- `column_names::Vector{Symbol}`: Data columns used by this formula
- `analysis::FormulaAnalysis`: Complete formula structure analysis

# Usage
```julia
compiled = compile_formula(model)
row_vec = Vector{Float64}(undef, compiled.output_width)

# Evaluate single row (zero allocations)
compiled(row_vec, data, 1)

# Access properties
println("Matrix width: ", length(compiled))
println("Uses columns: ", variables(compiled))
```
"""
struct CompiledFormula{H}
    formula_val::Val{H}
    output_width::Int
    column_names::Vector{Symbol}
    analysis::FormulaAnalysis
    
    # Inner constructor for type safety
    function CompiledFormula(formula_val::Val{H}, output_width::Int, 
                           column_names::Vector{Symbol}, 
                           analysis::FormulaAnalysis) where H
        new{H}(formula_val, output_width, column_names, analysis)
    end
end

###############################################################################
# Main Constructor Interface
###############################################################################

"""
    compile_formula(model) -> CompiledFormula

Compile a statistical model's formula into a zero-allocation evaluator.

This performs the complete 3-phase compilation pipeline:
1. **Phase 1**: Analyze formula structure
2. **Phase 2**: Generate optimized instructions  
3. **Phase 3**: Create @generated function

# Arguments
- `model`: A fitted statistical model (e.g., from GLM.jl)

# Returns
- `CompiledFormula`: A compiled formula that can evaluate model matrix rows

# Example
```julia
using GLM, DataFrames

df = DataFrame(x = randn(100), y = randn(100), group = rand(["A", "B"], 100))
model = lm(@formula(y ~ x + x^2 + group), df)

# Compile once (expensive ~1-10ms)
compiled = compile_formula(model)

# Use many times (fast ~50ns, zero allocations)
row_vec = Vector{Float64}(undef, length(compiled))
for i in 1:nrow(df)
    compiled(row_vec, Tables.columntable(df), i)
    # row_vec now contains model matrix row i
end
```
"""
function compile_formula(model)
    println("=== Complete Three-Phase Compilation ===")
    
    # Phase 1: Structure Analysis
    analysis = analyze_formula_structure(model)
    validate_analysis(analysis, model)
    
    # Phase 2: Instruction Generation  
    instructions = generate_instructions(analysis)
    
    # Phase 3: @generated Registration
    formula_hash = hash(string(fixed_effects_form(model).rhs))
    FORMULA_CACHE[formula_hash] = (instructions, analysis.all_columns, analysis.total_width)
    
    # Return struct directly
    return CompiledFormula(Val(formula_hash), analysis.total_width, analysis.all_columns, analysis)
end

###############################################################################
# Call Operator
###############################################################################

"""
    (compiled::CompiledFormula)(row_vec, data, row_idx)

Evaluate a single row of the model matrix into the provided vector.

# Arguments
- `row_vec::Vector{Float64}`: Pre-allocated output vector (length = compiled.output_width)
- `data`: Column-table format data (e.g., from `Tables.columntable(df)`)
- `row_idx::Int`: Which row to evaluate (1-based indexing)

# Performance
- **Time**: ~50-100 nanoseconds per call
- **Allocations**: Zero (uses pre-allocated `row_vec`)

# Example
```julia
compiled = compile_formula(model)
row_vec = Vector{Float64}(undef, length(compiled))
data = Tables.columntable(df)

compiled(row_vec, data, 5)  # Evaluate row 5
```
"""
@inline function (cf::CompiledFormula)(row_vec::AbstractVector{Float64}, data, row_idx::Int)
    # Delegate to the optimized @generated function
    modelrow!(row_vec, cf.formula_val, data, row_idx)
    return row_vec
end

###############################################################################
# Convenience Methods & Properties
###############################################################################

"""
    length(compiled::CompiledFormula) -> Int

Get the number of columns in the model matrix (same as `output_width`).
"""
Base.length(cf::CompiledFormula) = cf.output_width

"""
    size(compiled::CompiledFormula) -> Tuple{Int}

Get the size of model matrix rows as a tuple `(width,)`.
"""
Base.size(cf::CompiledFormula) = (cf.output_width,)

"""
    variables(compiled::CompiledFormula) -> Vector{Symbol}

Get the data column names used by this formula.
"""
variables(cf::CompiledFormula) = cf.column_names

"""
    formula_hash(compiled::CompiledFormula) -> UInt64

Get the unique hash identifying this compiled formula.
"""
formula_hash(cf::CompiledFormula{H}) where H = H

###############################################################################
# Batch Evaluation Methods
###############################################################################

"""
    evaluate_batch!(matrix, compiled, data, row_indices)

Evaluate multiple rows efficiently into a pre-allocated matrix.

# Arguments
- `matrix::Matrix{Float64}`: Pre-allocated output matrix (size: length(row_indices) × length(compiled))
- `compiled::CompiledFormula`: Compiled formula
- `data`: Column-table format data
- `row_indices`: Vector of row indices to evaluate

# Example
```julia
compiled = compile_formula(model)
matrix = Matrix{Float64}(undef, 100, length(compiled))
data = Tables.columntable(df)

evaluate_batch!(matrix, compiled, data, 1:100)
```
"""
function evaluate_batch!(matrix::AbstractMatrix{Float64}, cf::CompiledFormula, 
                        data, row_indices)
    @assert size(matrix, 2) == cf.output_width "Matrix width must match compiled formula width"
    @assert size(matrix, 1) >= length(row_indices) "Matrix height insufficient for row indices"
    
    # Use view to avoid allocations
    for (i, row_idx) in enumerate(row_indices)
        row_view = view(matrix, i, :)
        cf(row_view, data, row_idx)
    end
    
    return matrix
end

"""
    evaluate_batch(compiled, data, row_indices) -> Matrix{Float64}

Evaluate multiple rows and return a new matrix (allocating version).

Returns a matrix of size `length(row_indices) × length(compiled)`.
"""
function evaluate_batch(cf::CompiledFormula, data, row_indices)
    matrix = Matrix{Float64}(undef, length(row_indices), cf.output_width)
    evaluate_batch!(matrix, cf, data, row_indices)
    return matrix
end

###############################################################################
# Integration with Existing Pipeline
###############################################################################

# We need to modify phase3_generated_integration.jl to cache the analysis
const ANALYSIS_CACHE = Dict{UInt64, FormulaAnalysis}()

"""
    get_cached_analysis(formula_val::Val{H}) -> FormulaAnalysis

Retrieve the cached formula analysis for a compiled formula.
"""
function get_cached_analysis(formula_val::Val{H}) where H
    if haskey(ANALYSIS_CACHE, H)
        return ANALYSIS_CACHE[H]
    else
        error("Formula analysis not found in cache for hash $H. This shouldn't happen with proper compilation.")
    end
end

"""
    cache_analysis!(formula_hash::UInt64, analysis::FormulaAnalysis)

Cache the formula analysis during compilation.
This should be called from compile_formula_complete.
"""
function cache_analysis!(formula_hash::UInt64, analysis::FormulaAnalysis)
    ANALYSIS_CACHE[formula_hash] = analysis
end

###############################################################################
# Display and Debugging
###############################################################################

function Base.show(io::IO, cf::CompiledFormula{H}) where H
    print(io, "CompiledFormula{$H}(")
    print(io, "width=$(cf.output_width), ")
    print(io, "columns=$(cf.column_names))")
end

function Base.show(io::IO, ::MIME"text/plain", cf::CompiledFormula{H}) where H
    println(io, "CompiledFormula with hash $H:")
    println(io, "  Output width: $(cf.output_width)")
    println(io, "  Data columns: $(cf.column_names)")
    println(io, "  Terms: $(length(cf.analysis.terms))")
    
    # Show term breakdown
    for (i, term) in enumerate(cf.analysis.terms)
        println(io, "    $i. $(term.term_type) [$(term.start_position):$(term.start_position + term.width - 1)] $(term.term)")
    end
end

###############################################################################
# Advanced Methods for Margins.jl Integration
###############################################################################

"""
    get_variable_columns(compiled, variable::Symbol) -> Vector{Int}

Get the column indices affected by a specific variable.
Useful for marginal effects calculations.
"""
function get_variable_columns(cf::CompiledFormula, variable::Symbol)
    return get_variable_columns_flat(cf.analysis.position_map, variable)
end

"""
    get_variable_terms(compiled, variable::Symbol) -> Vector{TermAnalysis}

Get all terms that depend on a specific variable.
"""
function get_variable_terms(cf::CompiledFormula, variable::Symbol)
    terms = TermAnalysis[]
    for term in cf.analysis.terms
        if variable in term.columns_used
            push!(terms, term)
        end
    end
    return terms
end

# Export the main interface
export CompiledFormula, compile_formula, variables, evaluate_batch!, evaluate_batch
export get_variable_columns, get_variable_terms
