# ============================================================================
# modelmatrix!.jl - Basic efficient model matrix construction
# ============================================================================

"""
Extract design matrix from fitted model using standard interface

OPTIMIZED: Better error messages and type stability.
"""
function extract_model_matrix(model, df)
    existing_X = modelmatrix(model)
    n_model, n_df = size(existing_X, 1), nrow(df)
    
    if n_model != n_df
        throw(DimensionMismatch(
            "Model matrix has $n_model rows but data has $n_df rows"
        ))
    end
    
    return existing_X
end

"""
    modelmatrix!(X, rhs, data) -> X

Update matrix `X` in-place with the design matrix for `rhs` applied to `data`.
This is the core efficient operation that avoids allocating new matrices.

# Example
```julia
X = Matrix{Float64}(undef, nrow(df), ncols)
modelmatrix!(X, formula.rhs, df)  # X now contains the design matrix
```
"""
function modelmatrix!(
    X::AbstractMatrix,
    rhs,
    data;
)
    Tables.istable(data) || throw(ArgumentError("`data` is not Tables-compatible"))
    modelcols!(X, rhs, data)
    return X
end

"""
    modelcols!(dest, rhs, data) -> dest

Internal workhorse function that handles the actual copying from StatsModels.modelcols.
"""
function modelcols!(dest::AbstractMatrix, rhs, data)
    # Let StatsModels build the columns
    matrix_parts = StatsModels.modelcols(rhs, data)

    # More efficient normalization
    final_matrix = if matrix_parts isa Tuple
        reduce(hcat, matrix_parts)  # More efficient than hcat(matrix_parts...)
    elseif matrix_parts isa AbstractVector
        reshape(matrix_parts, :, 1)
    else
        matrix_parts
    end

    # Validate dimensions first
    dest_size = size(dest)
    final_size = size(final_matrix)
    if dest_size != final_size
        throw(DimensionMismatch(
            "Destination matrix is size $dest_size, but StatsModels created a matrix of size $final_size"
        ))
    end

    # Use copyto! for better performance than broadcasting
    copyto!(dest, final_matrix)
    return dest
end
