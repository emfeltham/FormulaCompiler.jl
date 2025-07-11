# standardized.jl
# Add this to _cols!.jl in EfficientModelMatrices.jl

"""
    _cols!(t::ZScoredTerm, d, X, j, ipm, fn_i, int_i) -> Int

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
- `ipm, fn_i, int_i`  
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
