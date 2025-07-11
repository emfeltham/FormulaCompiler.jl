# modelmatrix!_add.jl - COMPLETE REPLACEMENT

###############################################################################
# Selective Model Matrix Construction - FIXED VERSION
###############################################################################

"""
    modelmatrix_selective!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix, 
                          changed_variables::Vector{Symbol}, mapping::ColumnMapping)

FIXED: Update model matrix X by re-evaluating only columns affected by changed_variables.
"""
function modelmatrix_selective!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix, 
                               changed_variables::Vector{Symbol}, mapping::ColumnMapping)
    if isempty(changed_variables)
        return X
    end
    
    rhs = fixed_effects_form(ipm.model).rhs
    @assert width(rhs) == size(X, 2) "Matrix has wrong number of columns for model"
    validate_column_mapping(mapping, X)
    
    eval_columns_for_variables!(changed_variables, data, X, mapping, ipm)
    return X
end

"""
    modelmatrix_columns!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix, 
                        target_columns::Vector{Int}, mapping::ColumnMapping)

FIXED: Update specific columns of the model matrix X.
"""
function modelmatrix_columns!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix, 
                             target_columns::Vector{Int}, mapping::ColumnMapping)
    if isempty(target_columns)
        return X
    end
    
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
    
    fn_i = Ref(1)
    int_i = Ref(1)
    
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

FIXED: Update model matrix X with selective updates and memory sharing.
"""
function modelmatrix_with_base!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix,
                               X_base::AbstractMatrix, changed_variables::Vector{Symbol}, 
                               mapping::ColumnMapping)
    size(X) == size(X_base) || throw(DimensionMismatch(
        "Target and base matrices must have same dimensions"
    ))
    
    if isempty(changed_variables)
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
    
    # Start with base matrix
    X .= X_base
    
    # Update only the changed columns
    if !isempty(changed_cols)
        eval_columns_for_variables!(changed_variables, data, X, mapping, ipm)
    end
    
    return X
end
