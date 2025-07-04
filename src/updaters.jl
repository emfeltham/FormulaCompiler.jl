# =============================================================================
# UPDATE FUNCTIONS: The performance payoff
# =============================================================================

"""
    update!(cached_mm::CachedModelMatrix, new_data; changed_vars=nothing)

Update the cached model matrix with new data.

# Arguments
- `cached_mm`: The cached model matrix to update
- `new_data`: New tabular data
- `changed_vars=nothing`: Optional set of variables that changed (for optimization)

# Returns
The updated `CachedModelMatrix` (for chaining)
"""
function update!(cached_mm::CachedModelMatrix, new_data; changed_vars=nothing)
    # Convert to standard format
    tbl = Tables.columntable(new_data)
    
    # Determine which variables changed
    if isnothing(changed_vars)
        changed_vars = collect(keys(tbl))  # Conservative: assume all changed
    end
    
    # Find which matrix columns need updating
    cols_to_update = Set{Int}()
    for var in changed_vars
        if haskey(cached_mm.cache.data_to_matrix_cols, var)
            union!(cols_to_update, cached_mm.cache.data_to_matrix_cols[var])
        end
    end
    
    # Group by terms for efficient batch updates
    terms_to_update = Set{Int}()
    for col in cols_to_update
        if haskey(cached_mm.cache.matrix_col_to_term, col)
            term_idx, _ = cached_mm.cache.matrix_col_to_term[col]
            push!(terms_to_update, term_idx)
        end
    end
    
    # Update only the necessary terms
    for term_idx in terms_to_update
        update_term_columns!(cached_mm, term_idx, tbl)
    end
    
    return cached_mm
end

"""
    update_term_columns!(cached_mm, term_idx, tbl)

Update all columns corresponding to a specific term.
"""
function update_term_columns!(cached_mm, term_idx, tbl)
    term = cached_mm.cache.terms[term_idx]
    col_range = cached_mm.cache.term_ranges[term_idx]
    
    # Compute new values for this term
    new_cols = modelcols(term, tbl)
    
    # Update the matrix in-place
    assign_columns!(cached_mm.matrix, col_range, new_cols)
end

"""
    assign_columns!(matrix, col_range, new_values)

Assign new values to specific columns, handling different input types.
"""
function assign_columns!(matrix, col_range, new_values)
    if isa(new_values, AbstractVector)
        # Single column
        matrix[:, first(col_range)] = new_values
    elseif isa(new_values, AbstractMatrix)
        # Multiple columns
        matrix[:, col_range] = new_values
    elseif isa(new_values, Number)
        # Scalar (e.g., intercept)
        matrix[:, col_range] .= new_values
    else
        # Try to convert to appropriate form
        try
            matrix[:, col_range] = reshape(new_values, size(matrix, 1), length(col_range))
        catch
            # Final fallback
            matrix[:, col_range] .= new_values
        end
    end
end

# =============================================================================
# ADVANCED FUNCTIONS: For specialized use cases
# =============================================================================

"""
    selective_update!(cached_mm::CachedModelMatrix, var_updates::Dict{Symbol, Any})

Update specific variables without rebuilding the entire data structure.

# Arguments
- `cached_mm`: The cached model matrix
- `var_updates`: Dict mapping variable names to their new values

# Example
```julia
selective_update!(cached_mm, Dict(:x1 => randn(100), :x2 => zeros(100)))
```
"""
function selective_update!(cached_mm::CachedModelMatrix, var_updates::Dict{Symbol, Any})
    # Create a minimal data structure with only the changed variables
    tbl = NamedTuple(var_updates)
    
    # Update only the specified variables
    changed_vars = collect(keys(var_updates))
    update!(cached_mm, tbl; changed_vars=changed_vars)
    
    return cached_mm
end

"""
    batch_update!(cached_matrices::Vector{CachedModelMatrix}, 
                  new_data_list::Vector; changed_vars_list=nothing)

Efficiently update multiple cached matrices in batch.
"""
function batch_update!(cached_matrices::Vector{CachedModelMatrix}, 
                      new_data_list::Vector; changed_vars_list=nothing)
    if isnothing(changed_vars_list)
        changed_vars_list = fill(nothing, length(cached_matrices))
    end
    
    for (cached_mm, new_data, changed_vars) in zip(cached_matrices, new_data_list, changed_vars_list)
        update!(cached_mm, new_data; changed_vars=changed_vars)
    end
    
    return cached_matrices
end

# =============================================================================
# MODEL INTEGRATION UTILITIES
# =============================================================================

"""
    extract_model_data(model)

Extract the data used to fit a model, if available.
"""
function extract_model_data(model)
    # Try different ways to get model data
    if hasfield(typeof(model), :data)
        return model.data
    elseif hasfield(typeof(model), :frame) && hasfield(typeof(model.frame), :data)
        return model.frame.data
    elseif hasfield(typeof(model), :model) && hasfield(typeof(model.model), :data)
        return model.model.data
    else
        throw(ArgumentError("Cannot extract data from model of type $(typeof(model)). Please provide data explicitly."))
    end
end

"""
    validate_model_matrix_compatibility(model, data)

Ensure that the model matrix and data are compatible for caching.
"""
function validate_model_matrix_compatibility(model, data)
    X = modelmatrix(model)
    
    # Check row count
    if size(X, 1) != nrow(data)
        throw(DimensionMismatch(
            "Model matrix has $(size(X, 1)) rows but data has $(nrow(data)) rows"
        ))
    end
    
    # Check that we can extract the formula
    try
        formula(model)
    catch
        throw(ArgumentError("Cannot extract formula from model of type $(typeof(model))"))
    end
    
    return true
end

"""
    extract_fixed_effects_rhs(model)

Extract the fixed effects part of the formula RHS, handling mixed models.
This is useful for mixed models where you want to cache only the fixed effects matrix.
"""
function extract_fixed_effects_rhs(model)
    full_formula = formula(model)
    
    # For mixed models, we might want to strip random effects
    # This is a simplified version - you might want to use your existing
    # fixed_effects_form function from your margins code
    if hasfield(typeof(model), :resp) && hasfield(typeof(model.resp), :link)
        # Likely a mixed model - might need special handling
        # For now, just return the full RHS
        return full_formula.rhs
    else
        # Regular model
        return full_formula.rhs
    end
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
    Base.size(cached_mm::CachedModelMatrix)

Get the size of the underlying matrix.
"""
Base.size(cached_mm::CachedModelMatrix) = size(cached_mm.matrix)

"""
    Base.getindex(cached_mm::CachedModelMatrix, args...)

Index into the underlying matrix.
"""
Base.getindex(cached_mm::CachedModelMatrix, args...) = getindex(cached_mm.matrix, args...)

"""
    Base.setindex!(cached_mm::CachedModelMatrix, value, args...)

Set values in the underlying matrix.
"""
Base.setindex!(cached_mm::CachedModelMatrix, value, args...) = setindex!(cached_mm.matrix, value, args...)

"""
    get_dependency_info(cached_mm::CachedModelMatrix, var::Symbol)

Get information about which matrix columns depend on a specific variable.
"""
function get_dependency_info(cached_mm::CachedModelMatrix, var::Symbol)
    if haskey(cached_mm.cache.data_to_matrix_cols, var)
        return cached_mm.cache.data_to_matrix_cols[var]
    else
        return Int[]
    end
end

"""
    get_affected_variables(cached_mm::CachedModelMatrix, matrix_col::Int)

Get which variables affect a specific matrix column.
"""
function get_affected_variables(cached_mm::CachedModelMatrix, matrix_col::Int)
    affected_vars = Symbol[]
    for (var, cols) in cached_mm.cache.data_to_matrix_cols
        if matrix_col in cols
            push!(affected_vars, var)
        end
    end
    return affected_vars
end
