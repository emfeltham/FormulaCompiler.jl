# =============================================================================
# DEPENDENCY ANALYSIS: The core optimization (FIXED VERSION)
# =============================================================================

"""
    build_dependency_cache(schema_rhs, reference_data) -> TermDependencyCache

Analyze the formula structure and build a cache of dependency relationships.
This is the expensive operation that we do once and reuse many times.
"""
function build_dependency_cache(schema_rhs, reference_data)
    # Extract terms, filtering out random effects if present
    terms = collect_terms(schema_rhs)
    
    data_to_matrix_cols = Dict{Symbol, Vector{Int}}()
    matrix_col_to_term = Dict{Int, Tuple{Int, UnitRange{Int}}}()
    term_ranges = UnitRange{Int}[]
    term_widths = Int[]
    
    col_offset = 1
    for (term_idx, term) in enumerate(terms)
        term_width = compute_term_width(term, reference_data)
        push!(term_widths, term_width)
        
        col_range = col_offset:(col_offset + term_width - 1)
        push!(term_ranges, col_range)
        
        dependent_vars = extract_term_variables(term)
        
        for var in dependent_vars
            if !haskey(data_to_matrix_cols, var)
                data_to_matrix_cols[var] = Int[]
            end
            append!(data_to_matrix_cols[var], col_range)
        end
        
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
FIXED: Now handles Tuple types containing both MatrixTerm and RandomEffectsTerm (for MixedModels).
"""
function collect_terms(schema_rhs)
    if isa(schema_rhs, MatrixTerm)
        return collect(schema_rhs.terms)
    elseif isa(schema_rhs, Tuple)
        # Handle MixedModels case: (MatrixTerm, RandomEffectsTerm)
        # We only want the fixed effects terms for model matrix caching
        terms = AbstractTerm[]
        for component in schema_rhs
            if isa(component, MatrixTerm)
                append!(terms, collect(component.terms))
            # Skip RandomEffectsTerm - we only cache fixed effects matrix
            end
        end
        return terms
    else
        return [schema_rhs]
    end
end

"""
    extract_term_variables(term::AbstractTerm) -> Vector{Symbol}

Recursively extract all variable names that a term depends on.
FIXED: Now handles Tuple types and is more robust with missing fields.
"""
function extract_term_variables(term::AbstractTerm)
    vars = Symbol[]
    extract_term_variables!(vars, term)
    return unique(vars)
end

# Handle Tuple case (for MixedModels compatibility)
function extract_term_variables(term::Tuple)
    vars = Symbol[]
    for component in term
        if isa(component, AbstractTerm)
            extract_term_variables!(vars, component)
        end
    end
    return unique(vars)
end

function extract_term_variables!(vars::Vector{Symbol}, term::AbstractTerm)
    # Handle common term types with safer field access
    try
        if hasfield(typeof(term), :sym) && isdefined(term, :sym) && isa(term.sym, Symbol)
            push!(vars, term.sym)
        elseif hasfield(typeof(term), :terms) && isdefined(term, :terms)
            for subterm in term.terms
                extract_term_variables!(vars, subterm)
            end
        elseif hasfield(typeof(term), :args) && isdefined(term, :args)
            for arg in term.args
                if isa(arg, AbstractTerm)
                    extract_term_variables!(vars, arg)
                elseif isa(arg, Symbol)
                    push!(vars, arg)
                end
            end
        end
    catch e
        # Graceful fallback for unknown term types
        @debug "Could not extract variables from term of type $(typeof(term)): $e"
    end
end

"""
    compute_term_width(term, reference_data) -> Int

Compute how many columns a term generates in the model matrix.
FIXED: Better error handling and fallback for problematic terms.
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
    catch e
        # Fallback for problematic terms with better error handling
        @debug "modelcols failed for term $(typeof(term)): $e. Using fallback."
        try
            return estimate_term_width_fallback(term, tbl)
        catch e2
            @debug "Fallback also failed for term $(typeof(term)): $e2. Using width=1."
            return 1
        end
    end
end

"""
    estimate_term_width_fallback(term, tbl) -> Int

Fallback width estimation for when modelcols fails.
FIXED: More robust handling of different term types and data types.
"""
function estimate_term_width_fallback(term, tbl)
    try
        if hasfield(typeof(term), :sym) && isdefined(term, :sym) && haskey(tbl, term.sym)
            col_data = tbl[term.sym]
            if isa(col_data, CategoricalArray)
                return max(1, length(levels(col_data)) - 1)
            elseif eltype(col_data) <: Union{AbstractString, Symbol}
                # Handle string/symbol columns as categorical
                unique_vals = unique(col_data)
                return max(1, length(unique_vals) - 1)
            elseif eltype(col_data) <: Bool
                # Boolean variables get dummy coded
                return 1
            else
                # Continuous variables
                return 1
            end
        elseif hasfield(typeof(term), :terms) && isdefined(term, :terms)
            # For interactions, multiply subterm widths
            width = 1
            for subterm in term.terms
                width *= estimate_term_width_fallback(subterm, tbl)
            end
            return width
        else
            return 1
        end
    catch e
        @debug "Fallback width estimation failed: $e"
        return 1
    end
end
