# =============================================================================
# DEPENDENCY ANALYSIS: The core optimization
# =============================================================================

"""
    build_dependency_cache(schema_rhs, reference_data) -> TermDependencyCache

Analyze the formula structure and build a cache of dependency relationships.
This is the expensive operation that we do once and reuse many times.
"""
function build_dependency_cache(schema_rhs, reference_data)
    # Extract terms from the schema RHS
    terms = collect_terms(schema_rhs)
    
    # Initialize tracking structures
    data_to_matrix_cols = Dict{Symbol, Vector{Int}}()
    matrix_col_to_term = Dict{Int, Tuple{Int, UnitRange{Int}}}()
    term_ranges = UnitRange{Int}[]
    term_widths = Int[]
    
    # Analyze each term
    col_offset = 1
    for (term_idx, term) in enumerate(terms)
        # Compute term width using actual data
        term_width = compute_term_width(term, reference_data)
        push!(term_widths, term_width)
        
        # Column range for this term
        col_range = col_offset:(col_offset + term_width - 1)
        push!(term_ranges, col_range)
        
        # Find which data variables this term depends on
        dependent_vars = extract_term_variables(term)
        
        # Update dependency mappings
        for var in dependent_vars
            if !haskey(data_to_matrix_cols, var)
                data_to_matrix_cols[var] = Int[]
            end
            append!(data_to_matrix_cols[var], col_range)
        end
        
        # Map matrix columns back to terms
        for col in col_range
            matrix_col_to_term[col] = (term_idx, col_range)
        end
        
        col_offset += term_width
    end
    
    return TermDependencyCache(
        data_to_matrix_cols, matrix_col_to_term, 
        terms, term_ranges, term_widths
    )
end

"""
    collect_terms(schema_rhs) -> Vector{AbstractTerm}

Extract all terms from a schema-applied RHS, handling both single terms and MatrixTerm.
"""
function collect_terms(schema_rhs)
    if isa(schema_rhs, MatrixTerm)
        return collect(schema_rhs.terms)
    else
        return [schema_rhs]
    end
end

"""
    extract_term_variables(term::AbstractTerm) -> Vector{Symbol}

Recursively extract all variable names that a term depends on.
"""
function extract_term_variables(term::AbstractTerm)
    vars = Symbol[]
    extract_term_variables!(vars, term)
    return unique(vars)
end

function extract_term_variables!(vars::Vector{Symbol}, term::AbstractTerm)
    # Handle common term types
    if hasfield(typeof(term), :sym) && isa(term.sym, Symbol)
        push!(vars, term.sym)
    elseif hasfield(typeof(term), :terms)
        for subterm in term.terms
            extract_term_variables!(vars, subterm)
        end
    elseif hasfield(typeof(term), :args)
        for arg in term.args
            if isa(arg, AbstractTerm)
                extract_term_variables!(vars, arg)
            end
        end
    end
end

"""
    compute_term_width(term, reference_data) -> Int

Compute how many columns a term generates in the model matrix.
"""
function compute_term_width(term, reference_data)
    # Convert to Tables.jl format for consistency
    tbl = Tables.columntable(reference_data)
    
    try
        # Use modelcols to get actual width
        cols = modelcols(term, tbl)
        if isa(cols, AbstractVector)
            return 1
        elseif isa(cols, AbstractMatrix)
            return size(cols, 2)
        else
            return 1
        end
    catch
        # Fallback for problematic terms
        return estimate_term_width_fallback(term, tbl)
    end
end

"""
    estimate_term_width_fallback(term, tbl) -> Int

Fallback width estimation for when modelcols fails.
"""
function estimate_term_width_fallback(term, tbl)
    if hasfield(typeof(term), :sym) && haskey(tbl, term.sym)
        col_data = tbl[term.sym]
        if isa(col_data, CategoricalArray)
            return max(1, length(levels(col_data)) - 1)
        else
            unique_vals = unique(col_data)
            return max(1, length(unique_vals) - 1)
        end
    elseif hasfield(typeof(term), :terms)
        # For interactions, multiply subterm widths
        width = 1
        for subterm in term.terms
            width *= estimate_term_width_fallback(subterm, tbl)
        end
        return width
    else
        return 1
    end
end
