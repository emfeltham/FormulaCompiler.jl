# ============================================================================
# modelmatrix!.jl - Basic efficient model matrix construction
# ============================================================================

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

    # Normalize the result to always be a single matrix
    final_matrix = if matrix_parts isa Tuple
        hcat(matrix_parts...)
    elseif matrix_parts isa AbstractVector
        reshape(matrix_parts, :, 1)
    else
        matrix_parts
    end

    # Validate and copy
    if size(dest) != size(final_matrix)
        throw(DimensionMismatch(
            "Destination matrix is size $(size(dest)), but StatsModels created a matrix of size $(size(final_matrix))"
        ))
    end

    dest .= final_matrix
    return dest
end
