# matrix_reuse_helpers.jl - CLEAN REPLACEMENT

"""
Extract design matrix from fitted model using standard interface
"""
function extract_model_matrix(model, df)
    existing_X = modelmatrix(model)
    @assert size(existing_X, 1) == nrow(df) "Model matrix rows ($(size(existing_X, 1))) don't match data rows ($(nrow(df)))"
    return existing_X
end

# ============================================================================
# dependencies.jl - Analyze which terms depend on which variables
# ============================================================================

"""
    MatrixUpdatePlan

Pre-computed information about which terms in a formula depend on which variables
and where those terms appear in the design matrix.

# Fields
- `var_to_terms::Dict{Symbol, Vector{Tuple{Int, UnitRange{Int}}}}`: Maps each variable to the terms it affects and their column ranges
- `term_widths::Vector{Int}`: Width (number of columns) of each term
- `total_width::Int`: Total number of columns in the design matrix
"""
struct MatrixUpdatePlan
    var_to_terms::Dict{Symbol, Vector{Tuple{Int, UnitRange{Int}}}}
    term_widths::Vector{Int}
    total_width::Int
end

"""
    analyze_dependencies(rhs, variables, data) -> MatrixUpdatePlan
    analyze_dependencies(rhs, data) -> MatrixUpdatePlan

Analyze which terms in the formula `rhs` depend on which `variables`.
If `variables` is not provided, analyzes all variables in the data.

This pre-computation allows for efficient selective updates of the design matrix.

# Example
```julia
plan = analyze_dependencies(formula.rhs, [:x1, :x2], df)
# Now you can efficiently update only parts of the matrix that depend on x1 or x2
```
"""
function analyze_dependencies(rhs, variables::Vector{Symbol}, data)
    tbl = Tables.columntable(data)
    terms = _extract_terms(rhs)
    
    # Pre-compute term widths
    term_widths = [estimate_term_width(term, tbl) for term in terms]
    total_width = sum(term_widths)
    
    # Map variables to terms and column ranges
    var_to_terms = Dict{Symbol, Vector{Tuple{Int, UnitRange{Int}}}}()
    
    col_offset = 1
    for (i, term) in enumerate(terms)
        ncols = term_widths[i]
        col_range = col_offset:(col_offset + ncols - 1)
        
        # Find which variables affect this term
        affected_vars = find_variables_in_term(term, variables)
        for var in affected_vars
            if !haskey(var_to_terms, var)
                var_to_terms[var] = Tuple{Int, UnitRange{Int}}[]
            end
            push!(var_to_terms[var], (i, col_range))
        end
        
        col_offset += ncols
    end
    
    return MatrixUpdatePlan(var_to_terms, term_widths, total_width)
end

# Convenience method: analyze all variables
function analyze_dependencies(rhs, data)
    tbl = Tables.columntable(data)
    all_vars = collect(keys(tbl))
    return analyze_dependencies(rhs, all_vars, data)
end

"""
    estimate_term_width(term, data) -> Int

Estimate how many columns a term will produce in the design matrix
without actually constructing the term.
"""
function estimate_term_width(term, tbl)
    if isa(term, StatsModels.InterceptTerm)
        return 1
    elseif isa(term, StatsModels.ContinuousTerm)
        return 1
    elseif isa(term, StatsModels.CategoricalTerm)
        col_data = tbl[term.sym]
        if isa(col_data, CategoricalArray)
            return max(1, length(levels(col_data)) - 1)
        else
            return max(1, length(unique(col_data)) - 1)
        end
    elseif isa(term, StatsModels.InteractionTerm)
        width = 1
        for subterm in term.terms
            width *= estimate_term_width(subterm, tbl)
        end
        return width
    elseif hasfield(typeof(term), :sym) && isa(term.sym, Symbol)
        return 1  # Treat function terms as continuous
    else
        # Conservative fallback
        return 1
    end
end

"""
    find_variables_in_term(term, candidate_vars) -> Vector{Symbol}

Find which of the `candidate_vars` actually appear in the given term.
"""
function find_variables_in_term(term, candidate_vars::Vector{Symbol})
    if isa(term, StatsModels.ContinuousTerm)
        return term.sym in candidate_vars ? [term.sym] : Symbol[]
    elseif isa(term, StatsModels.InterceptTerm)
        return Symbol[]
    elseif isa(term, StatsModels.InteractionTerm)
        result = Symbol[]
        for subterm in term.terms
            append!(result, find_variables_in_term(subterm, candidate_vars))
        end
        return unique!(result)
    elseif hasfield(typeof(term), :sym) && isa(term.sym, Symbol)
        return term.sym in candidate_vars ? [term.sym] : Symbol[]
    else
        # Fallback: use StatsModels
        try
            term_vars = StatsModels.termvars(term)
            return intersect(candidate_vars, term_vars)
        catch
            return Symbol[]
        end
    end
end

# Helper function
_extract_terms(rhs) = isa(rhs, StatsModels.MatrixTerm) ? collect(rhs.terms) : [rhs]
