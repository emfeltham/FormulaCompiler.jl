# ADD TO EXISTING modelmatrix!.jl - Extensions for selective matrix construction

###############################################################################
# Selective Model Matrix Construction
###############################################################################

"""
    modelmatrix_selective!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix, 
                          changed_variables::Vector{Symbol}, mapping::ColumnMapping)

Update model matrix X by re-evaluating only columns affected by changed_variables.
All other columns remain unchanged (memory sharing where possible).

# Arguments
- `ipm`: InplaceModeler for the model
- `data`: New data (NamedTuple) with updated variable values
- `X`: Target matrix to update (modified in-place)
- `changed_variables`: Variables that have changed and need column updates
- `mapping`: ColumnMapping to determine which columns are affected

# Returns
- `X`: The same matrix passed in, now updated
"""
function modelmatrix_selective!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix, 
                               changed_variables::Vector{Symbol}, mapping::ColumnMapping)
    if isempty(changed_variables)
        return X  # No changes needed
    end
    
    # Validate that mapping matches the model and matrix
    rhs = fixed_effects_form(ipm.model).rhs
    @assert width(rhs) == size(X, 2) "Matrix has wrong number of columns for model"
    validate_column_mapping(mapping, X)
    
    # Update columns affected by the changed variables
    eval_columns_for_variables!(changed_variables, data, X, mapping, ipm)
    
    return X
end

"""
    modelmatrix_columns!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix, 
                        target_columns::Vector{Int}, mapping::ColumnMapping)

Update specific columns of the model matrix X.
More fine-grained control than modelmatrix_selective!.

# Arguments
- `ipm`: InplaceModeler for the model
- `data`: Data (NamedTuple) to evaluate
- `X`: Target matrix to update (modified in-place)
- `target_columns`: Specific column indices to update
- `mapping`: ColumnMapping for the model

# Returns
- `X`: The same matrix passed in, with specified columns updated
"""
function modelmatrix_columns!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix, 
                             target_columns::Vector{Int}, mapping::ColumnMapping)
    if isempty(target_columns)
        return X
    end
    
    # Validate inputs
    validate_column_mapping(mapping, X)
    max_col = maximum(target_columns)
    min_col = minimum(target_columns)
    @assert 1 ≤ min_col ≤ max_col ≤ size(X, 2) "Invalid column indices"
    
    # Find which terms correspond to the target columns
    terms_to_update = Set{AbstractTerm}()
    term_ranges = Dict{AbstractTerm, UnitRange{Int}}()
    
    for (term, range) in mapping.term_info
        if !isempty(range) && !isempty(intersect(collect(range), target_columns))
            push!(terms_to_update, term)
            term_ranges[term] = range
        end
    end
    
    # Initialize counters
    fn_i = Ref(1)
    int_i = Ref(1)
    
    # Update each relevant term
    for term in terms_to_update
        range = term_ranges[term]
        if !isempty(range)
            _cols_selective!(term, data, X, first(range), target_columns, ipm, fn_i, int_i)
        end
    end
    
    return X
end

"""
    modelmatrix_with_base!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix,
                          X_base::AbstractMatrix, changed_variables::Vector{Symbol}, 
                          mapping::ColumnMapping)

Update model matrix X by copying changed columns from data evaluation and 
sharing memory for unchanged columns from X_base.

# Arguments
- `ipm`: InplaceModeler for the model
- `data`: New data with updated variables
- `X`: Target matrix to update
- `X_base`: Base matrix with unchanged column data
- `changed_variables`: Variables that have changed
- `mapping`: ColumnMapping for the model

# Returns
- `X`: Updated matrix with selective column updates
"""
function modelmatrix_with_base!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix,
                               X_base::AbstractMatrix, changed_variables::Vector{Symbol}, 
                               mapping::ColumnMapping)
    # Validate dimensions
    size(X) == size(X_base) || throw(DimensionMismatch(
        "Target and base matrices must have same dimensions"
    ))
    
    if isempty(changed_variables)
        # No variables changed, just copy the base matrix
        X .= X_base
        return X
    end
    
    # Get affected and unaffected columns
    total_cols = size(X, 2)
    changed_cols = Set{Int}()
    
    for var in changed_variables
        var_cols = get_variable_columns_flat(mapping, var)
        union!(changed_cols, var_cols)
    end
    
    changed_cols = sort(collect(changed_cols))
    unchanged_cols = get_unchanged_columns(mapping, changed_variables, total_cols)
    
    # Update only the changed columns
    if !isempty(changed_cols)
        eval_columns_for_variables!(changed_variables, data, X, mapping, ipm)
    end
    
    # Share memory for unchanged columns
    update_matrix_columns!(X, X_base, changed_cols, unchanged_cols)
    
    return X
end
