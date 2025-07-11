# _cols!_add.jl - COMPLETE REPLACEMENT

###############################################################################
# Selective Column Evaluation Functions - FIXED VERSION
###############################################################################

"""
    _cols_selective!(term::AbstractTerm, d, X, j, affected_cols::Vector{Int}, 
                     ipm, fn_i, int_i) -> Int

FIXED: Selective version of _cols! that only updates columns in `affected_cols`.
Handles InteractionTerm and FunctionTerm correctly without BoundsError.
"""
function _cols_selective!(term::AbstractTerm, d, X, j, affected_cols::Vector{Int}, 
                         ipm, fn_i, int_i)
    w = width(term)
    
    if w == 0
        return j
    end
    
    term_cols = j:(j + w - 1)
    cols_to_update = intersect(term_cols, affected_cols)
    
    if isempty(cols_to_update)
        return j + w
    end
    
    # FIXED: Special handling for InteractionTerm and FunctionTerm
    if term isa InteractionTerm || term isa FunctionTerm
        # These terms need to work with full target matrix
        # Backup non-affected columns
        backup_cols = setdiff(collect(term_cols), cols_to_update)
        backup_data = Dict{Int, Vector{Float64}}()
        
        for col in backup_cols
            if 1 <= col <= size(X, 2)
                backup_data[col] = copy(X[:, col])
            end
        end
        
        # Evaluate directly into target matrix
        _cols!(term, d, X, j, ipm, fn_i, int_i)
        
        # Restore backed up columns
        for (col, values) in backup_data
            X[:, col] = values
        end
    else
        # Simple terms - optimized path
        if w == 1 && length(cols_to_update) == 1
            # Single column optimization
            col = cols_to_update[1]
            if term isa ContinuousTerm || term isa Term
                copy!(view(X, :, col), d[term.sym])
            elseif term isa ConstantTerm
                fill!(view(X, :, col), term.n)
            elseif term isa InterceptTerm{true}
                fill!(view(X, :, col), 1.0)
            elseif term isa InterceptTerm{false}
                # Should not happen in practice
                fill!(view(X, :, col), 0.0)
            else
                # Fallback to temp_matrix
                temp_col = Vector{Float64}(undef, size(X, 1))
                temp_matrix = reshape(temp_col, :, 1)
                _cols!(term, d, temp_matrix, 1, ipm, fn_i, int_i)
                X[:, col] = temp_col
            end
        else
            # Multi-column case
            temp_matrix = Matrix{Float64}(undef, size(X, 1), w)
            _cols!(term, d, temp_matrix, 1, ipm, fn_i, int_i)
            
            for col in cols_to_update
                local_col = col - j + 1
                if 1 <= local_col <= w
                    X[:, col] = temp_matrix[:, local_col]
                end
            end
        end
    end
    
    return j + w
end

"""
    eval_columns_for_variable!(variable::Symbol, data::NamedTuple, X::AbstractMatrix, 
                              mapping::ColumnMapping, ipm::InplaceModeler)

FIXED: High-level interface to update all columns affected by a single variable.
"""
function eval_columns_for_variable!(variable::Symbol, data::NamedTuple, X::AbstractMatrix, 
                                   mapping::ColumnMapping, ipm::InplaceModeler)
    var_term_ranges = get_variable_term_ranges(mapping, variable)
    
    if isempty(var_term_ranges)
        return
    end
    
    affected_cols = get_variable_columns_flat(mapping, variable)
    validate_column_mapping(mapping, X)
    
    fn_i = Ref(1)
    int_i = Ref(1)
    
    for (term, range) in var_term_ranges
        if !isempty(range)
            _cols_selective!(term, data, X, first(range), affected_cols, ipm, fn_i, int_i)
        end
    end
end

"""
    eval_columns_for_variables!(variables::Vector{Symbol}, data::NamedTuple, 
                               X::AbstractMatrix, mapping::ColumnMapping, 
                               ipm::InplaceModeler)

FIXED: Update all columns affected by multiple variables efficiently.
"""
function eval_columns_for_variables!(variables::Vector{Symbol}, data::NamedTuple, 
                                    X::AbstractMatrix, mapping::ColumnMapping, 
                                    ipm::InplaceModeler)
    if isempty(variables)
        return
    end
    
    all_affected_cols = Set{Int}()
    var_term_map = Dict{Symbol, Vector{Tuple{AbstractTerm, UnitRange{Int}}}}()
    
    for var in variables
        var_ranges = get_variable_term_ranges(mapping, var)
        var_term_map[var] = var_ranges
        
        var_cols = get_variable_columns_flat(mapping, var)
        union!(all_affected_cols, var_cols)
    end
    
    affected_cols = sort(collect(all_affected_cols))
    validate_column_mapping(mapping, X)
    
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

FIXED: Update target matrix with memory sharing for unchanged columns.
"""
function update_matrix_columns!(X_target::AbstractMatrix, X_base::AbstractMatrix,
                               changed_cols::Vector{Int}, unchanged_cols::Vector{Int})
    size(X_target) == size(X_base) || throw(DimensionMismatch(
        "Target and base matrices must have same dimensions"
    ))
    
    # Share memory for unchanged columns
    for col in unchanged_cols
        if 1 <= col <= size(X_target, 2)
            X_target[:, col] = view(X_base, :, col)
        end
    end
end
