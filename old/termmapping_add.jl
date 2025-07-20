# termmapping_add.jl
# extra functions for termmapping.jl


"""
    get_variable_columns_flat(mapping::ColumnMapping, var::Symbol) -> Vector{Int}

Get flat vector of all column indices where a variable appears.
More efficient than `get_all_variable_columns` for direct indexing.
"""
function get_variable_columns_flat(mapping::ColumnMapping, var::Symbol)
    ranges = get_variable_ranges(mapping, var)
    cols = Int[]
    for range in ranges
        append!(cols, collect(range))
    end
    return unique!(sort!(cols))
end

"""
    get_unchanged_columns(mapping::ColumnMapping, changed_vars::Vector{Symbol}, total_cols::Int) -> Vector{Int}

Return column indices that are NOT affected by any of the changed variables.
These columns can share memory during selective updates.
"""
function get_unchanged_columns(mapping::ColumnMapping, changed_vars::Vector{Symbol}, total_cols::Int)
    changed_cols = Set{Int}()
    
    for var in changed_vars
        var_cols = get_variable_columns_flat(mapping, var)
        union!(changed_cols, var_cols)
    end
    
    return [col for col in 1:total_cols if col ∉ changed_cols]
end

"""
    validate_column_mapping(mapping::ColumnMapping, X::AbstractMatrix)

Validate that the column mapping is consistent with the actual matrix dimensions.
Throws an error if mapping refers to non-existent columns.
"""
function validate_column_mapping(mapping::ColumnMapping, X::AbstractMatrix)
    matrix_cols = size(X, 2)
    
    if mapping.total_columns != matrix_cols
        throw(DimensionMismatch(
            "Column mapping expects $(mapping.total_columns) columns, " *
            "but matrix has $matrix_cols columns"
        ))
    end
    
    # Check that all referenced columns exist
    for (term, range) in mapping.term_info
        if !isempty(range) && (first(range) < 1 || last(range) > matrix_cols)
            throw(BoundsError(
                "Term $term references columns $range, " *
                "but matrix only has columns 1:$matrix_cols"
            ))
        end
    end
    
    return true
end

"""
    get_variable_term_ranges(mapping::ColumnMapping, var::Symbol) -> Vector{Tuple{AbstractTerm, UnitRange{Int}}}

Get all (term, range) pairs where a variable appears.
Useful for selective term evaluation.
"""
function get_variable_term_ranges(mapping::ColumnMapping, var::Symbol)
    result = Tuple{AbstractTerm, UnitRange{Int}}[]
    
    for (term, range) in mapping.term_info
        term_vars = collect_termvars_recursive(term)
        if var in term_vars
            push!(result, (term, range))
        end
    end
    
    return result
end

"""
    build_variable_term_map(mapping::ColumnMapping) -> Dict{Symbol, Vector{Tuple{AbstractTerm, UnitRange{Int}}}}

Pre-compute term-range mapping for all variables for efficient lookups.
"""
function build_variable_term_map(mapping::ColumnMapping)
    var_term_map = Dict{Symbol, Vector{Tuple{AbstractTerm, UnitRange{Int}}}}()
    
    # Get all variables from the mapping
    all_vars = Set{Symbol}()
    for (term, _) in mapping.term_info
        union!(all_vars, collect_termvars_recursive(term))
    end
    
    # Build mapping for each variable
    for var in all_vars
        var_term_map[var] = get_variable_term_ranges(mapping, var)
    end
    
    return var_term_map
end
