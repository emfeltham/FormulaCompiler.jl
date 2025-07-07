# standarized.jl

# Add this to _cols!.jl in EfficientModelMatrices.jl

"""
    _cols!(t::ZScoredTerm, d, X, j, _ipm, _fn_i, _int_i) -> Int

In-place writer for a Z-scored term, applying the transformation without allocations.

# Arguments

- `t::ZScoredTerm`  
  A Z-scored term wrapping an underlying term with center and scale values.
- `d::NamedTuple`  
  Column-table data mapping variable names to vectors.
- `X::AbstractMatrix`  
  The destination matrix in which to write starting at column `j`.
- `j::Int`  
  The index of the first column in `X` to fill.
- `_ipm, _fn_i, _int_i`  
  Parameters passed through to underlying term evaluation.

# Behavior

1. Delegates to the underlying term's `_cols!` method to fill the raw data.
2. Applies Z-score transformation in-place: `(x - center) / scale`.
3. Handles both scalar and vector center/scale values.

# Returns

- The next free column index after writing the Z-scored columns.
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
    n_cols = j_end - j_start + 1
    
    if center isa Number && scale isa Number
        # Scalar center and scale - apply to all columns
        # Optimize for center == 0 case (like StatsBase.jl does)
        if center == zero(center)
            inv_scale = inv(scale)
            @inbounds for col in j_start:j_end, row in axes(X, 1)
                X[row, col] = X[row, col] * inv_scale
            end
        else
            inv_scale = inv(scale)
            @inbounds for col in j_start:j_end, row in axes(X, 1)
                X[row, col] = (X[row, col] - center) * inv_scale
            end
        end
    elseif center isa AbstractVector && scale isa AbstractVector
        # Vector center and scale - apply column-wise
        @assert length(center) == n_cols "Center vector length must match number of columns"
        @assert length(scale) == n_cols "Scale vector length must match number of columns"
        
        @inbounds for (col_idx, col) in enumerate(j_start:j_end)
            c = center[col_idx]
            s = scale[col_idx]
            if c == zero(c)
                inv_s = inv(s)
                for row in axes(X, 1)
                    X[row, col] = X[row, col] * inv_s
                end
            else
                inv_s = inv(s)
                for row in axes(X, 1)
                    X[row, col] = (X[row, col] - c) * inv_s
                end
            end
        end
    elseif center isa Number && scale isa AbstractVector
        # Scalar center, vector scale
        @assert length(scale) == n_cols "Scale vector length must match number of columns"
        
        @inbounds for (col_idx, col) in enumerate(j_start:j_end)
            s = scale[col_idx]
            if center == zero(center)
                inv_s = inv(s)
                for row in axes(X, 1)
                    X[row, col] = X[row, col] * inv_s
                end
            else
                inv_s = inv(s)
                for row in axes(X, 1)
                    X[row, col] = (X[row, col] - center) * inv_s
                end
            end
        end
    elseif center isa AbstractVector && scale isa Number
        # Vector center, scalar scale
        @assert length(center) == n_cols "Center vector length must match number of columns"
        
        inv_scale = inv(scale)
        @inbounds for (col_idx, col) in enumerate(j_start:j_end)
            c = center[col_idx]
            if c == zero(c)
                for row in axes(X, 1)
                    X[row, col] = X[row, col] * inv_scale
                end
            else
                for row in axes(X, 1)
                    X[row, col] = (X[row, col] - c) * inv_scale
                end
            end
        end
    else
        error("Unsupported center/scale types: $(typeof(center)), $(typeof(scale))")
    end
end
