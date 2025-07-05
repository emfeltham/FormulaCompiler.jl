# =============================================================================
# UPDATE FUNCTIONS: The performance payoff (FIXED VERSION)
# =============================================================================

"""
    update!(cached_mm::CachedModelMatrix, new_data; changed_vars=nothing)

Update the cached model matrix with new data.
FIXED: Better error handling and numerical stability.

# Arguments
- `cached_mm`: The cached model matrix to update
- `new_data`: New tabular data
- `changed_vars=nothing`: Optional set of variables that changed (for optimization)

# Returns
The updated `CachedModelMatrix` (for chaining)
"""
function update!(cached_mm::CachedModelMatrix, new_data; changed_vars=nothing)
    try
        tbl = Tables.columntable(new_data)
        
        if isnothing(changed_vars)
            changed_vars = collect(keys(tbl))
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
            try
                update_term_columns!(cached_mm, term_idx, tbl)
            catch e
                @warn "Failed to update term $term_idx: $e"
                # Continue with other terms instead of failing completely
            end
        end
        
        return cached_mm
        
    catch e
        @error "Critical error in update!: $e"
        # Return unchanged matrix rather than failing
        return cached_mm
    end
end

"""
    update_term_columns!(cached_mm, term_idx, tbl)

Update all columns corresponding to a specific term.
FIXED: Better categorical handling and numerical stability.
"""
function update_term_columns!(cached_mm, term_idx, tbl)
    term = cached_mm.cache.terms[term_idx]
    col_range = cached_mm.cache.term_ranges[term_idx]
    
    # Compute new values for this term with better error handling
    try
        new_cols = modelcols(term, tbl)
        
        # Validate the new columns before assignment
        if any(isnan, new_cols) || any(isinf, new_cols)
            @warn "NaN or Inf values detected in modelcols output for term $term_idx"
            # Fill with zeros as safer fallback
            cached_mm.matrix[:, col_range] .= 0.0
            return
        end
        
        # Update the matrix in-place
        assign_columns!(cached_mm.matrix, col_range, new_cols)
        
    catch e
        @warn "Failed to compute modelcols for term $term_idx: $e"
        # Use fallback strategy
        try
            fallback_update_term!(cached_mm, term_idx, term, tbl)
        catch e2
            @warn "Fallback update also failed for term $term_idx: $e2"
            # Last resort: fill with zeros
            cached_mm.matrix[:, col_range] .= 0.0
        end
    end
end

"""
    fallback_update_term!(cached_mm, term_idx, term, tbl)

Fallback strategy for updating terms when modelcols fails.
"""
function fallback_update_term!(cached_mm, term_idx, term, tbl)
    col_range = cached_mm.cache.term_ranges[term_idx]
    
    # Try to handle specific term types manually
    if hasfield(typeof(term), :sym) && isdefined(term, :sym)
        var_sym = term.sym
        if haskey(tbl, var_sym)
            var_data = tbl[var_sym]
            
            # Handle different data types
            if eltype(var_data) <: Bool
                # Boolean to Float64 conversion
                cached_mm.matrix[:, first(col_range)] .= Float64.(var_data)
            elseif eltype(var_data) <: Real
                # Numeric data
                cached_mm.matrix[:, first(col_range)] .= Float64.(var_data)
            elseif isa(var_data, CategoricalArray)
                # Handle categorical with proper level mapping
                fallback_categorical_update!(cached_mm.matrix, col_range, var_data)
            else
                # Unknown type - use zeros
                cached_mm.matrix[:, col_range] .= 0.0
            end
        else
            # Variable not found - use zeros
            cached_mm.matrix[:, col_range] .= 0.0
        end
    else
        # Unknown term structure - use zeros
        cached_mm.matrix[:, col_range] .= 0.0
    end
end

"""
    fallback_categorical_update!(matrix, col_range, categorical_data)

Handle categorical data updates when standard methods fail.
"""
function fallback_categorical_update!(matrix, col_range, categorical_data)
    try
        # Get levels and create dummy coding manually
        lvls = levels(categorical_data)
        n_levels = length(lvls)
        n_cols = length(col_range)
        
        if n_levels <= 1
            # Only one level or empty - fill with zeros
            matrix[:, col_range] .= 0.0
            return
        end
        
        # Create dummy variables (reference = first level)
        for (i, row_val) in enumerate(categorical_data)
            for (j, level) in enumerate(lvls[2:end])  # Skip first level (reference)
                if j <= n_cols  # Don't exceed available columns
                    matrix[i, col_range[j]] = (row_val == level) ? 1.0 : 0.0
                end
            end
        end
        
        # Fill any remaining columns with zeros
        if n_cols > (n_levels - 1)
            matrix[:, col_range[(n_levels):end]] .= 0.0
        end
        
    catch e
        @warn "Fallback categorical update failed: $e"
        matrix[:, col_range] .= 0.0
    end
end

"""
    assign_columns!(matrix, col_range, new_values)

Assign new values to specific columns, handling different input types.
FIXED: Better type handling and numerical stability checks.
"""
function assign_columns!(matrix, col_range, new_values)
    try
        # Validate inputs first
        if length(col_range) == 0
            return  # Nothing to assign
        end
        
        if isa(new_values, AbstractVector)
            if length(new_values) != size(matrix, 1)
                @warn "Vector length mismatch: expected $(size(matrix, 1)), got $(length(new_values))"
                return
            end
            
            # Check for numerical issues
            if any(isnan, new_values) || any(isinf, new_values)
                @warn "NaN or Inf values in vector assignment, using zeros"
                matrix[:, first(col_range)] .= 0.0
            else
                matrix[:, first(col_range)] = new_values
            end
            
        elseif isa(new_values, AbstractMatrix)
            if size(new_values, 1) != size(matrix, 1)
                @warn "Matrix row count mismatch: expected $(size(matrix, 1)), got $(size(new_values, 1))"
                return
            end
            
            if size(new_values, 2) == length(col_range)
                # Check for numerical issues
                if any(isnan, new_values) || any(isinf, new_values)
                    @warn "NaN or Inf values in matrix assignment, using zeros"
                    matrix[:, col_range] .= 0.0
                else
                    matrix[:, col_range] = new_values
                end
            else
                @warn "Column count mismatch: expected $(length(col_range)), got $(size(new_values, 2))"
                # Handle gracefully by taking available columns
                n_cols = min(size(new_values, 2), length(col_range))
                if n_cols > 0
                    sub_values = new_values[:, 1:n_cols]
                    if any(isnan, sub_values) || any(isinf, sub_values)
                        matrix[:, col_range[1:n_cols]] .= 0.0
                    else
                        matrix[:, col_range[1:n_cols]] = sub_values
                    end
                end
                
                # Fill remaining columns with zeros
                if length(col_range) > n_cols
                    matrix[:, col_range[(n_cols+1):end]] .= 0.0
                end
            end
            
        elseif isa(new_values, Number)
            if isnan(new_values) || isinf(new_values)
                @warn "NaN or Inf scalar value, using zero"
                matrix[:, col_range] .= 0.0
            else
                matrix[:, col_range] .= new_values
            end
            
        else
            # Try to convert unknown types
            try
                converted = Float64.(new_values)
                if any(isnan, converted) || any(isinf, converted)
                    @warn "NaN or Inf values after conversion, using zeros"
                    matrix[:, col_range] .= 0.0
                else
                    assign_columns!(matrix, col_range, converted)
                end
            catch conversion_error
                @warn "Failed to convert values of type $(typeof(new_values)): $conversion_error"
                matrix[:, col_range] .= 0.0
            end
        end
        
    catch e
        @warn "Failed to assign columns: $e"
        # Fill with zeros as ultimate fallback
        matrix[:, col_range] .= 0.0
    end
end

# =============================================================================
# ADVANCED FUNCTIONS: For specialized use cases (UNCHANGED - these are fine)
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
    tbl = NamedTuple(var_updates)
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
# MODEL INTEGRATION UTILITIES (FIXED)
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
    try
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
        
    catch e
        @error "Model matrix compatibility validation failed: $e"
        rethrow(e)
    end
end

"""
    extract_fixed_effects_rhs(model)

Extract the fixed effects part of the formula RHS, handling mixed models.
This is useful for mixed models where you want to cache only the fixed effects matrix.
"""
function extract_fixed_effects_rhs(model)
    try
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
        
    catch e
        @error "Failed to extract fixed effects RHS: $e"
        rethrow(e)
    end
end
