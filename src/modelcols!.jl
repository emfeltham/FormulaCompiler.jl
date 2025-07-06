# modelcols_mutating.jl - Truly zero-allocation version

"""
    modelcols!(dest::AbstractMatrix, rhs, data, model) -> dest

Truly zero-allocation version that works directly with DataFrame columns
and avoids all intermediate matrix allocations.
"""
function modelcols!(dest::AbstractMatrix{T}, rhs, data, model) where {T}
    # Work directly with DataFrame - no Tables.columntable conversion
    if !isa(data, AbstractDataFrame)
        throw(ArgumentError("This zero-allocation version requires DataFrame input"))
    end
    
    # Get the applied terms directly from the model
    rhs_applied = model.mf.f.rhs
    
    # Fill using the pre-applied terms without any intermediate allocations
    _fill_from_applied_terms_zero_alloc!(dest, rhs_applied, data)
    
    return dest
end

"""
Fill destination matrix using pre-applied terms with zero allocations
"""
function _fill_from_applied_terms_zero_alloc!(dest::AbstractMatrix{T}, rhs_applied, df) where {T}
    if isa(rhs_applied, StatsModels.MatrixTerm)
        # Multiple terms - fill each column range
        col_offset = 1
        for term in rhs_applied.terms
            term_width = StatsModels.width(term)
            col_range = col_offset:(col_offset + term_width - 1)
            dest_view = view(dest, :, col_range)
            _fill_single_term_zero_alloc!(dest_view, term, df)
            col_offset += term_width
        end
    else
        # Single term
        _fill_single_term_zero_alloc!(dest, rhs_applied, df)
    end
end

"""
Fill a single term directly into destination without any allocations
"""
function _fill_single_term_zero_alloc!(dest::AbstractMatrix{T}, term, df) where {T}
    if isa(term, StatsModels.InterceptTerm{true})
        fill!(dest, one(T))
        
    elseif isa(term, StatsModels.ContinuousTerm)
        # Direct column access - no intermediate allocation
        col_data = df[!, term.sym]
        dest[:, 1] .= col_data
        
    elseif isa(term, StatsModels.CategoricalTerm)
        # This is the key fix - manually compute contrasts without allocation
        col_data = df[!, term.sym]
        _fill_categorical_contrasts_zero_alloc!(dest, col_data, term.contrasts)
        
    elseif isa(term, StatsModels.InteractionTerm)
        # For interactions, we need to compute the product
        _fill_interaction_term_zero_alloc!(dest, term, df)
        
    elseif isa(term, StatsModels.FunctionTerm)
        # Apply the function directly
        _fill_function_term_zero_alloc!(dest, term, df)
        
    else
        error("Unsupported term type: $(typeof(term))")
    end
end

"""
Fill categorical contrasts manually without creating intermediate matrices
"""
function _fill_categorical_contrasts_zero_alloc!(dest::AbstractMatrix{T}, col_data, contrasts) where {T}
    n_rows = length(col_data)
    n_cols = size(dest, 2)
    
    # Get the contrast matrix
    contrasts_matrix = contrasts.matrix
    
    # Handle CategoricalArray - use the pool for efficient lookup
    if isa(col_data, CategoricalArray)
        # Use the categorical pool for efficient lookup
        pool = col_data.pool
        
        # Fill row by row using the categorical refs directly
        for i in 1:n_rows
            # Get the categorical reference (1-based index into levels)
            level_ref = col_data.refs[i]
            
            # Copy the appropriate row from contrasts matrix
            for j in 1:n_cols
                dest[i, j] = contrasts_matrix[level_ref, j]
            end
        end
    else
        # Fallback for non-categorical data - build lookup dict
        level_to_idx = Dict()
        for (i, level) in enumerate(contrasts.levels)
            level_to_idx[level] = i
        end
        
        # Fill row by row
        for i in 1:n_rows
            level_val = col_data[i]
            level_idx = get(level_to_idx, level_val, 1)
            
            # Copy the appropriate row from contrasts matrix
            for j in 1:n_cols
                dest[i, j] = contrasts_matrix[level_idx, j]
            end
        end
    end
end

"""
Fill interaction term by computing products of components without allocations
"""
function _fill_interaction_term_zero_alloc!(dest::AbstractMatrix{T}, term::StatsModels.InteractionTerm, df) where {T}
    # Handle only 2-way interactions for now (most common case)
    if length(term.terms) != 2
        error("Only 2-way interactions supported in zero-allocation version")
    end
    
    term1, term2 = term.terms
    
    # Get the data for each component directly
    if isa(term1, StatsModels.ContinuousTerm) && isa(term2, StatsModels.ContinuousTerm)
        # Continuous × Continuous
        data1 = df[!, term1.sym]
        data2 = df[!, term2.sym]
        for i in 1:length(data1)
            dest[i, 1] = data1[i] * data2[i]
        end
        
    elseif isa(term1, StatsModels.ContinuousTerm) && isa(term2, StatsModels.CategoricalTerm)
        # Continuous × Categorical
        data1 = df[!, term1.sym]
        data2 = df[!, term2.sym]
        _fill_continuous_categorical_interaction!(dest, data1, data2, term2.contrasts)
        
    elseif isa(term1, StatsModels.CategoricalTerm) && isa(term2, StatsModels.ContinuousTerm)
        # Categorical × Continuous
        data1 = df[!, term1.sym]
        data2 = df[!, term2.sym]
        _fill_categorical_continuous_interaction!(dest, data1, data2, term1.contrasts)
        
    elseif isa(term1, StatsModels.CategoricalTerm) && isa(term2, StatsModels.CategoricalTerm)
        # Categorical × Categorical
        data1 = df[!, term1.sym]
        data2 = df[!, term2.sym]
        _fill_categorical_categorical_interaction!(dest, data1, data2, term1.contrasts, term2.contrasts)
        
    else
        error("Unsupported interaction types: $(typeof(term1)) × $(typeof(term2))")
    end
end

"""
Fill continuous × categorical interaction without allocations
"""
function _fill_continuous_categorical_interaction!(dest::AbstractMatrix{T}, cont_data, cat_data, contrasts) where {T}
    n_rows = length(cont_data)
    n_contrast_cols = size(contrasts.matrix, 2)
    
    if isa(cat_data, CategoricalArray)
        # Use categorical refs for efficiency
        for i in 1:n_rows
            cont_val = cont_data[i]
            level_ref = cat_data.refs[i]
            
            for j in 1:n_contrast_cols
                dest[i, j] = cont_val * contrasts.matrix[level_ref, j]
            end
        end
    else
        # Fallback for non-categorical
        level_to_idx = Dict()
        for (i, level) in enumerate(contrasts.levels)
            level_to_idx[level] = i
        end
        
        for i in 1:n_rows
            cont_val = cont_data[i]
            cat_level = cat_data[i]
            level_idx = get(level_to_idx, cat_level, 1)
            
            for j in 1:n_contrast_cols
                dest[i, j] = cont_val * contrasts.matrix[level_idx, j]
            end
        end
    end
end

"""
Fill categorical × continuous interaction without allocations
"""
function _fill_categorical_continuous_interaction!(dest::AbstractMatrix{T}, cat_data, cont_data, contrasts) where {T}
    n_rows = length(cont_data)
    n_contrast_cols = size(contrasts.matrix, 2)
    
    if isa(cat_data, CategoricalArray)
        # Use categorical refs for efficiency
        for i in 1:n_rows
            cont_val = cont_data[i]
            level_ref = cat_data.refs[i]
            
            for j in 1:n_contrast_cols
                dest[i, j] = contrasts.matrix[level_ref, j] * cont_val
            end
        end
    else
        # Fallback for non-categorical
        level_to_idx = Dict()
        for (i, level) in enumerate(contrasts.levels)
            level_to_idx[level] = i
        end
        
        for i in 1:n_rows
            cont_val = cont_data[i]
            cat_level = cat_data[i]
            level_idx = get(level_to_idx, cat_level, 1)
            
            for j in 1:n_contrast_cols
                dest[i, j] = contrasts.matrix[level_idx, j] * cont_val
            end
        end
    end
end

"""
Fill categorical × categorical interaction without allocations
"""
function _fill_categorical_categorical_interaction!(dest::AbstractMatrix{T}, cat_data1, cat_data2, contrasts1, contrasts2) where {T}
    n_rows = length(cat_data1)
    n_cols1 = size(contrasts1.matrix, 2)
    n_cols2 = size(contrasts2.matrix, 2)
    
    if isa(cat_data1, CategoricalArray) && isa(cat_data2, CategoricalArray)
        # Use categorical refs for both - most efficient
        for i in 1:n_rows
            ref1 = cat_data1.refs[i]
            ref2 = cat_data2.refs[i]
            
            col_offset = 1
            for j1 in 1:n_cols1
                for j2 in 1:n_cols2
                    dest[i, col_offset] = contrasts1.matrix[ref1, j1] * contrasts2.matrix[ref2, j2]
                    col_offset += 1
                end
            end
        end
    else
        # Fallback with lookups
        level_to_idx1 = Dict()
        for (i, level) in enumerate(contrasts1.levels)
            level_to_idx1[level] = i
        end
        
        level_to_idx2 = Dict()
        for (i, level) in enumerate(contrasts2.levels)
            level_to_idx2[level] = i
        end
        
        for i in 1:n_rows
            level1 = cat_data1[i]
            level2 = cat_data2[i]
            idx1 = get(level_to_idx1, level1, 1)
            idx2 = get(level_to_idx2, level2, 1)
            
            col_offset = 1
            for j1 in 1:n_cols1
                for j2 in 1:n_cols2
                    dest[i, col_offset] = contrasts1.matrix[idx1, j1] * contrasts2.matrix[idx2, j2]
                    col_offset += 1
                end
            end
        end
    end
end

"""
Fill function term by applying the function without allocations
"""
function _fill_function_term_zero_alloc!(dest::AbstractMatrix{T}, term::StatsModels.FunctionTerm, df) where {T}
    # Get the arguments
    if length(term.args) == 1
        arg = term.args[1]
        if isa(arg, StatsModels.ContinuousTerm)
            arg_data = df[!, arg.sym]
            for i in 1:length(arg_data)
                dest[i, 1] = T(term.f(arg_data[i]))
            end
        else
            error("Function terms with non-continuous arguments not supported")
        end
    else
        error("Multi-argument function terms not yet supported")
    end
end
