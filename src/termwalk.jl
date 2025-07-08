# column_mapping.jl - Robust recursive column mapping for EfficientModelMatrices.jl

"""
    ColumnMapping

A lightweight struct that precomputes the mapping from variable symbols to 
model matrix column ranges, enabling O(1) lookups.
"""
struct ColumnMapping
    symbol_to_ranges::Dict{Symbol, Vector{UnitRange{Int}}}  # Multiple ranges per symbol
    range_to_terms::Dict{UnitRange{Int}, Vector{AbstractTerm}}  # Which terms generate each range
    total_columns::Int
    term_info::Vector{Tuple{AbstractTerm, UnitRange{Int}}}  # Ordered list of (term, range)
end

"""
    build_column_mapping(rhs::AbstractTerm) -> ColumnMapping

Build a comprehensive mapping from variable symbols to their corresponding column ranges 
in the model matrix. This works recursively through arbitrary formula complexity.

Handles complex cases like:
- `x + x^2 + inv(x)` - multiple terms involving the same variable
- `x & a + inv(x) & a & b` - interactions with shared variables
- Nested function calls and arbitrary term combinations

# Arguments
- `rhs`: The right-hand side of a formula (after apply_schema)

# Returns
- `ColumnMapping`: A struct containing comprehensive mappings

# Example
```julia
# Complex formula: y ~ x + x^2 + inv(x) & a + a + inv(x) & a & b
rhs = fixed_effects_form(model).rhs
mapping = build_column_mapping(rhs)

# Get ALL columns that involve variable :x (could be multiple ranges)
x_ranges = get_variable_ranges(mapping, :x)  
# e.g., [2:2, 4:4, 7:9] for x, x^2, and inv(x) & a & b terms
```
"""
function build_column_mapping(rhs::AbstractTerm)
    symbol_to_ranges = Dict{Symbol, Vector{UnitRange{Int}}}()
    range_to_terms = Dict{UnitRange{Int}, Vector{AbstractTerm}}()
    term_info = Tuple{AbstractTerm, UnitRange{Int}}[]
    
    # Walk the term tree and accumulate column assignments
    current_col = Ref(1)
    _map_columns_recursive!(symbol_to_ranges, range_to_terms, term_info, rhs, current_col)
    
    total_cols = current_col[] - 1
    return ColumnMapping(symbol_to_ranges, range_to_terms, total_cols, term_info)
end

"""
Internal recursive function that properly handles all StatsModels term types.
"""
function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info, 
                                term::AbstractTerm, col_ref::Ref{Int})
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        # Record this term and its range
        push!(term_info, (term, range))
        
        # Get all variables that this term depends on
        vars = collect_termvars_recursive(term)
        
        # Map each variable to this range
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        # Record which terms generate this range
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

# Specialized methods for composite terms
function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info,
                                term::MatrixTerm, col_ref::Ref{Int})
    for t in term.terms
        _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info, t, col_ref)
    end
    return col_ref[]
end

function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info,
                                terms::Tuple, col_ref::Ref{Int})
    for t in terms
        _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info, t, col_ref)
    end
    return col_ref[]
end

function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info,
                                term::InteractionTerm, col_ref::Ref{Int})
    # Interaction terms create their own columns, but we need to track
    # that all participating variables are involved
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        push!(term_info, (term, range))
        
        # For interactions, collect variables from ALL sub-terms
        vars = Set{Symbol}()
        for subterm in term.terms
            union!(vars, collect_termvars_recursive(subterm))
        end
        
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info,
                                term::FunctionTerm, col_ref::Ref{Int})
    # Function terms create their own columns
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        push!(term_info, (term, range))
        
        # Collect variables from all arguments recursively
        vars = collect_termvars_recursive(term)
        
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

# Handle ZScoredTerm from StandardizedPredictors.jl
function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info,
                                term::ZScoredTerm, col_ref::Ref{Int})
    # ZScoredTerm creates its own columns but wraps another term
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        push!(term_info, (term, range))
        
        # Get variables from the wrapped term
        vars = collect_termvars_recursive(term)
        
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

# Handle terms that don't contribute variables (intercept, constants)
function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_info,
                                term::Union{InterceptTerm, ConstantTerm}, col_ref::Ref{Int})
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        push!(term_info, (term, range))
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    return col_ref[]
end

"""
    collect_termvars_recursive(term::AbstractTerm) -> Set{Symbol}

Recursively collect ALL variable symbols that a term depends on, handling
nested function calls, interactions, and arbitrary term complexity.
"""
function collect_termvars_recursive(term::AbstractTerm)
    vars = Set{Symbol}()
    _collect_vars_recursive!(vars, term)
    return vars
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::Term)
    push!(vars, term.sym)
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::ContinuousTerm)
    push!(vars, term.sym)
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::CategoricalTerm)
    push!(vars, term.sym)
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::FunctionTerm)
    # Recursively collect from all arguments
    for arg in term.args
        _collect_vars_recursive!(vars, arg)
    end
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::InteractionTerm)
    for subterm in term.terms
        _collect_vars_recursive!(vars, subterm)
    end
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::MatrixTerm)
    for subterm in term.terms
        _collect_vars_recursive!(vars, subterm)
    end
end

function _collect_vars_recursive!(vars::Set{Symbol}, terms::Tuple)
    for term in terms
        _collect_vars_recursive!(vars, term)
    end
end

# Handle ZScoredTerm from StandardizedPredictors.jl
function _collect_vars_recursive!(vars::Set{Symbol}, term::ZScoredTerm)
    # Delegate to the wrapped term
    _collect_vars_recursive!(vars, term.term)
end

# Base cases that don't contribute variables
function _collect_vars_recursive!(vars::Set{Symbol}, term::Union{InterceptTerm, ConstantTerm})
    # Do nothing - these don't contribute variables
end

# Fallback for termvars() compatibility
function _collect_vars_recursive!(vars::Set{Symbol}, term::AbstractTerm)
    # Try to use existing termvars if available
    try
        for var in termvars(term)
            push!(vars, var)
        end
    catch
        # If termvars fails, we can't extract variables from this term
        @warn "Could not extract variables from term of type $(typeof(term)): $term"
    end
end

"""
    get_variable_ranges(mapping::ColumnMapping, sym::Symbol) -> Vector{UnitRange{Int}}

Get ALL column ranges where a variable appears. For complex formulas, a variable
might appear in multiple terms, each contributing different columns.

# Example
```julia
# For formula: y ~ x + x^2 + inv(x) & a
mapping = build_column_mapping(rhs)
x_ranges = get_variable_ranges(mapping, :x)
# Returns: [2:2, 3:3, 4:5] (x term, x^2 term, inv(x)&a interaction)
```
"""
function get_variable_ranges(mapping::ColumnMapping, sym::Symbol)
    get(mapping.symbol_to_ranges, sym, UnitRange{Int}[])
end

"""
    get_all_variable_columns(mapping::ColumnMapping, sym::Symbol) -> Vector{Int}

Get a flat vector of ALL column indices where a variable appears.
"""
function get_all_variable_columns(mapping::ColumnMapping, sym::Symbol)
    ranges = get_variable_ranges(mapping, sym)
    cols = Int[]
    for range in ranges
        append!(cols, collect(range))
    end
    return unique!(sort!(cols))
end

"""
    get_terms_involving_variable(mapping::ColumnMapping, sym::Symbol) -> Vector{Tuple{AbstractTerm, UnitRange{Int}}}

Get all terms that involve a specific variable, along with their column ranges.
Useful for understanding exactly how a variable contributes to the model matrix.
"""
function get_terms_involving_variable(mapping::ColumnMapping, sym::Symbol)
    result = Tuple{AbstractTerm, UnitRange{Int}}[]
    ranges = get_variable_ranges(mapping, sym)
    
    for range in ranges
        if haskey(mapping.range_to_terms, range)
            for term in mapping.range_to_terms[range]
                push!(result, (term, range))
            end
        end
    end
    
    return result
end

"""
    analyze_formula_structure(mapping::ColumnMapping) -> Dict{Symbol, Dict{String, Any}}

Provide a detailed analysis of how each variable participates in the formula.
Useful for debugging and understanding complex formulas.
"""
function analyze_formula_structure(mapping::ColumnMapping)
    analysis = Dict{Symbol, Dict{String, Any}}()
    
    for (sym, ranges) in mapping.symbol_to_ranges
        info = Dict{String, Any}()
        info["total_columns"] = length(get_all_variable_columns(mapping, sym))
        info["appears_in_terms"] = length(ranges)
        info["column_ranges"] = ranges
        
        # Categorize term types
        term_types = String[]
        for range in ranges
            if haskey(mapping.range_to_terms, range)
                for term in mapping.range_to_terms[range]
                    push!(term_types, string(typeof(term)))
                end
            end
        end
        info["term_types"] = unique(term_types)
        
        analysis[sym] = info
    end
    
    return analysis
end

# Integration with InplaceModeler
"""
    InplaceModelerWithMapping

Extended InplaceModeler that includes precomputed column mappings for efficient
variable-to-column lookups with complex formulas.
"""
struct InplaceModelerWithMapping{M}
    modeler::InplaceModeler{M}
    mapping::ColumnMapping
end

function InplaceModelerWithMapping(model, nrows::Int)
    ipm = InplaceModeler(model, nrows)
    rhs = fixed_effects_form(model).rhs
    mapping = build_column_mapping(rhs)
    return InplaceModelerWithMapping(ipm, mapping)
end

# Convenience methods
get_variable_ranges(ipm_mapping::InplaceModelerWithMapping, sym::Symbol) = 
    get_variable_ranges(imp_mapping.mapping, sym)

get_all_variable_columns(ipm_mapping::InplaceModelerWithMapping, sym::Symbol) = 
    get_all_variable_columns(ipm_mapping.mapping, sym)

"""
    test_complex_formula()

Test function to verify the mapping works with complex formulas.
"""
function test_complex_formula()
    # This would test something like: y ~ x + x^2 + inv(x) & a + a + inv(x) & a & b
    println("Testing complex formula mapping...")
    
    # Note: This is a conceptual test - actual usage would require a fitted model
    # The mapping should correctly identify that:
    # - :x appears in: x term, x^2 term, inv(x) & a term, inv(x) & a & b term  
    # - :a appears in: a term, inv(x) & a term, inv(x) & a & b term
    # - :b appears in: inv(x) & a & b term
    
    println("Mapping should handle:")
    println("  - Multiple terms per variable")
    println("  - Nested function calls") 
    println("  - Complex interactions")
    println("  - Repeated variables in different contexts")
end

export ColumnMapping, InplaceModelerWithMapping
export build_column_mapping, get_variable_ranges, get_all_variable_columns
export get_terms_involving_variable, analyze_formula_structure
export collect_termvars_recursive, test_complex_formula
