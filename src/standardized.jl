# standardized.jl
# Add this to _cols!.jl in EfficientModelMatrices.jl

"""
    _cols!(t::ZScoredTerm, d, X, j, ipm, fn_i, int_i) -> Int

In-place writer for a Z-scored term, applying the transformation without allocations.
"""
function _cols!(t::ZScoredTerm, d, X, j, ipm, fn_i, int_i)
    # First, fill the columns using the underlying term
    j_next = _cols!(t.term, d, X, j, ipm, fn_i, int_i)
    
    # Apply Z-score transformation in-place to the columns we just filled
    apply_zscore_inplace!(X, j, j_next - 1, t.center, t.scale)
    
    return j_next
end

"""
    apply_zscore_inplace!(X, j_start, j_end, center, scale)

Apply Z-score transformation in-place to columns j_start:j_end of matrix X.
Handles both scalar and vector center/scale values efficiently.
"""
function apply_zscore_inplace!(X::AbstractMatrix, j_start::Int, j_end::Int, center, scale)
    # Extract scalar values from center and scale
    c = center isa Number ? center : center[1]  # Handle both scalar and vector
    s = scale isa Number ? scale : scale[1]
    
    # Apply transformation efficiently
    if c == 0
        inv_s = 1.0 / s
        @inbounds for col in j_start:j_end, row in axes(X, 1)
            X[row, col] = X[row, col] * inv_s
        end
    else
        inv_s = 1.0 / s
        @inbounds for col in j_start:j_end, row in axes(X, 1)
            X[row, col] = (X[row, col] - c) * inv_s
        end
    end
end

function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info,
                                term::ZScoredTerm, col_ref::Ref{Int})
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        push!(term_info, (term, range))
        term_to_range[term] = range
        
        # Collect variables from the wrapped term
        vars = collect_termvars_recursive(term.term)
        
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::ZScoredTerm)
    # Delegate to the wrapped term
    _collect_vars_recursive!(vars, term.term)
end

function _evaluate_term_full!(term::ZScoredTerm, data::NamedTuple, output::AbstractMatrix)
    # Evaluate underlying term first
    _evaluate_term_full!(term.term, data, output)
    
    # Apply Z-score transformation to all columns
    _apply_zscore_transform!(output, term.center, term.scale)
    
    return output
end

function _evaluate_single_column_direct!(term::ZScoredTerm, data::NamedTuple, output::AbstractVector)
    @assert width(term) == 1 "ZScoredTerm direct evaluation only for width=1"
    
    # Evaluate underlying term first
    _evaluate_single_column_direct!(term.term, data, output)
    
    # Apply Z-score transformation in-place
    _apply_zscore_single_column!(output, term.center, term.scale)
    
    return output
end

"""
    _apply_zscore_single_column!(output::AbstractVector, center, scale)

Apply Z-score transformation to a single column vector in-place.
"""
function _apply_zscore_single_column!(output::AbstractVector, center, scale)
    if center isa Number && scale isa Number
        if center == 0
            inv_scale = 1.0 / scale
            @inbounds @simd for i in 1:length(output)
                output[i] *= inv_scale
            end
        else
            inv_scale = 1.0 / scale
            @inbounds @simd for i in 1:length(output)
                output[i] = (output[i] - center) * inv_scale
            end
        end
    elseif center isa AbstractVector && scale isa AbstractVector
        # This shouldn't happen for single column, but handle gracefully
        @assert length(center) == 1 && length(scale) == 1 "Vector center/scale for single column must have length 1"
        c, s = center[1], scale[1]
        if c == 0
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(output)
                output[i] *= inv_s
            end
        else
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(output)
                output[i] = (output[i] - c) * inv_s
            end
        end
    elseif center isa Number && scale isa AbstractVector
        @assert length(scale) == 1 "Vector scale for single column must have length 1"
        s = scale[1]
        if center == 0
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(output)
                output[i] *= inv_s
            end
        else
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(output)
                output[i] = (output[i] - center) * inv_s
            end
        end
    elseif center isa AbstractVector && scale isa Number
        @assert length(center) == 1 "Vector center for single column must have length 1"
        c = center[1]
        if c == 0
            inv_scale = 1.0 / scale
            @inbounds @simd for i in 1:length(output)
                output[i] *= inv_scale
            end
        else
            inv_scale = 1.0 / scale
            @inbounds @simd for i in 1:length(output)
                output[i] = (output[i] - c) * inv_scale
            end
        end
    else
        error("Unsupported center/scale types for Z-score: $(typeof(center)), $(typeof(scale))")
    end
    
    return output
end

"""
    _apply_zscore_transform!(output::AbstractMatrix, center, scale)

Apply Z-score transformation to a matrix in-place, handling various center/scale combinations.
"""
function _apply_zscore_transform!(output::AbstractMatrix, center, scale)
    n_rows, n_cols = size(output)
    
    if center isa Number && scale isa Number
        # Scalar center and scale - apply to all columns
        if center == 0
            inv_scale = 1.0 / scale
            @inbounds for i in eachindex(output)
                output[i] *= inv_scale
            end
        else
            inv_scale = 1.0 / scale
            @inbounds for i in eachindex(output)
                output[i] = (output[i] - center) * inv_scale
            end
        end
    elseif center isa AbstractVector && scale isa AbstractVector
        # Vector center and scale - apply column-wise
        @assert length(center) == n_cols "Center vector length must match number of columns"
        @assert length(scale) == n_cols "Scale vector length must match number of columns"
        
        @inbounds for col in 1:n_cols
            c = center[col]
            s = scale[col]
            if c == 0
                inv_s = 1.0 / s
                for row in 1:n_rows
                    output[row, col] *= inv_s
                end
            else
                inv_s = 1.0 / s
                for row in 1:n_rows
                    output[row, col] = (output[row, col] - c) * inv_s
                end
            end
        end
    elseif center isa Number && scale isa AbstractVector
        # Scalar center, vector scale
        @assert length(scale) == n_cols "Scale vector length must match number of columns"
        
        @inbounds for col in 1:n_cols
            s = scale[col]
            if center == 0
                inv_s = 1.0 / s
                for row in 1:n_rows
                    output[row, col] *= inv_s
                end
            else
                inv_s = 1.0 / s
                for row in 1:n_rows
                    output[row, col] = (output[row, col] - center) * inv_s
                end
            end
        end
    elseif center isa AbstractVector && scale isa Number
        # Vector center, scalar scale
        @assert length(center) == n_cols "Center vector length must match number of columns"
        
        inv_scale = 1.0 / scale
        @inbounds for col in 1:n_cols
            c = center[col]
            if c == 0
                for row in 1:n_rows
                    output[row, col] *= inv_scale
                end
            else
                for row in 1:n_rows
                    output[row, col] = (output[row, col] - c) * inv_scale
                end
            end
        end
    else
        error("Unsupported center/scale types for Z-score: $(typeof(center)), $(typeof(scale))")
    end
    
    return output
end

