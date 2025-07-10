# ADD TO EXISTING _cols!.jl - Extensions for selective column evaluation

###############################################################################
# Selective Column Evaluation Functions
###############################################################################

"""
    _cols_selective!(term::AbstractTerm, d, X, j, affected_cols::Vector{Int}, 
                     ipm, fn_i, int_i) -> Int

Selective version of _cols! that only updates columns in `affected_cols`.
Columns not in `affected_cols` are left unchanged.

This is used when we know that only certain columns need updating due to
a variable change, allowing memory sharing for unchanged columns.
"""
function _cols_selective!(term::AbstractTerm, d, X, j, affected_cols::Vector{Int}, 
                         ipm, fn_i, int_i)
    # Get the width of this term
    w = width(term)
    
    if w == 0
        return j
    end
    
    # Determine which columns this term would write to
    term_cols = j:(j + w - 1)
    
    # Find intersection with affected columns
    cols_to_update = intersect(collect(term_cols), affected_cols)
    
    if isempty(cols_to_update)
        # No columns need updating for this term, skip
        return j + w
    end
    
    # Create a temporary matrix to hold the full term evaluation
    temp_matrix = Matrix{Float64}(undef, size(X, 1), w)
    
    # Evaluate the full term into temporary matrix
    _cols!(term, d, temp_matrix, 1, ipm, fn_i, int_i)
    
    # Copy only the affected columns to the target matrix
    for col in cols_to_update
        local_col = col - j + 1  # Column index within this term
        X[:, col] = temp_matrix[:, local_col]
    end
    
    return j + w
end

"""
    eval_columns_for_variable!(variable::Symbol, data::NamedTuple, X::AbstractMatrix, 
                              mapping::ColumnMapping, ipm::InplaceModeler)

High-level interface to update all columns affected by a single variable.
Uses the column mapping to determine which terms need re-evaluation.
"""
function eval_columns_for_variable!(variable::Symbol, data::NamedTuple, X::AbstractMatrix, 
                                   mapping::ColumnMapping, ipm::InplaceModeler)
    # Get all terms and ranges that involve this variable
    var_term_ranges = get_variable_term_ranges(mapping, variable)
    
    if isempty(var_term_ranges)
        return  # Variable doesn't affect any terms
    end
    
    # Get all affected columns
    affected_cols = get_variable_columns_flat(mapping, variable)
    
    # Validate dimensions
    validate_column_mapping(mapping, X)
    
    # Initialize counters for _cols! dispatch
    fn_i = Ref(1)
    int_i = Ref(1)
    
    # Process each term that involves this variable
    for (term, range) in var_term_ranges
        if !isempty(range)
            # Update only the affected columns for this term
            _cols_selective!(term, data, X, first(range), affected_cols, ipm, fn_i, int_i)
        end
    end
end

"""
    eval_columns_for_variables!(variables::Vector{Symbol}, data::NamedTuple, 
                               X::AbstractMatrix, mapping::ColumnMapping, 
                               ipm::InplaceModeler)

Update all columns affected by any of the specified variables.
More efficient than calling eval_columns_for_variable! multiple times
when multiple variables change simultaneously.
"""
function eval_columns_for_variables!(variables::Vector{Symbol}, data::NamedTuple, 
                                    X::AbstractMatrix, mapping::ColumnMapping, 
                                    ipm::InplaceModeler)
    if isempty(variables)
        return
    end
    
    # Get all affected columns across all variables
    all_affected_cols = Set{Int}()
    var_term_map = Dict{Symbol, Vector{Tuple{AbstractTerm, UnitRange{Int}}}}()
    
    for var in variables
        var_ranges = get_variable_term_ranges(mapping, var)
        var_term_map[var] = var_ranges
        
        var_cols = get_variable_columns_flat(mapping, var)
        union!(all_affected_cols, var_cols)
    end
    
    affected_cols = sort(collect(all_affected_cols))
    
    # Validate dimensions
    validate_column_mapping(mapping, X)
    
    # Initialize counters
    fn_i = Ref(1)
    int_i = Ref(1)
    
    # Process each unique term that needs updating
    processed_terms = Set{AbstractTerm}()
    
    for var in variables
        for (term, range) in var_term_map[var]
            if term ∉ processed_terms && !isempty(range)
                _cols_selective!(term, data, X, first(range), affected_cols, ipm, fn_i, int_i)
                push!(processed_terms, term)
            end
        end
    end
end

"""
    update_matrix_columns!(X_target::AbstractMatrix, X_base::AbstractMatrix,
                          changed_cols::Vector{Int}, unchanged_cols::Vector{Int})

Update target matrix by copying changed columns with new values and 
sharing memory for unchanged columns.

# Arguments
- `X_target`: Matrix to update (must be pre-allocated)
- `X_base`: Base matrix with unchanged column data
- `changed_cols`: Column indices that have been updated in X_target
- `unchanged_cols`: Column indices to copy from X_base (memory sharing)
"""
function update_matrix_columns!(X_target::AbstractMatrix, X_base::AbstractMatrix,
                               changed_cols::Vector{Int}, unchanged_cols::Vector{Int})
    # Validate dimensions
    size(X_target) == size(X_base) || throw(DimensionMismatch(
        "Target and base matrices must have same dimensions"
    ))
    
    # Share memory for unchanged columns
    for col in unchanged_cols
        X_target[:, col] = view(X_base, :, col)
    end
    
    # Note: changed columns should already be updated in X_target
    # This function just handles the memory sharing for unchanged columns
end
