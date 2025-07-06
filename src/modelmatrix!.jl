using Tables, StatsModels

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
This avoids allocating new matrices.
"""
function modelmatrix!(X::AbstractMatrix{T}, rhs, data) where T
    Tables.istable(data) || throw(ArgumentError("`data` is not Tables-compatible"))
    return modelcols!(X, rhs, data)
end

"""
    modelcols!(dest, rhs, data) -> dest

Internal workhorse: copies StatsModels.modelcols blocks directly into `dest` without temporaries.
"""
function modelcols!(dest::AbstractMatrix{T}, rhs, data) where T
    Tables.istable(data) || throw(ArgumentError("`data` is not Tables-compatible"))

    # Let StatsModels produce its parts
    parts_raw = StatsModels.modelcols(rhs, data)

    # Normalize into a tuple of blocks
    blocks = if parts_raw isa Tuple
        parts_raw
    elseif parts_raw isa AbstractMatrix || parts_raw isa AbstractVector
        (parts_raw,)
    else
        throw(ArgumentError("Unsupported return type from StatsModels.modelcols: $(typeof(parts_raw))"))
    end

    # Compute total number of columns
    total_cols = 0
    for block in blocks
        total_cols += block isa AbstractVector ? 1 : size(block, 2)
    end

    # Validate dimensions
    n_rows, n_cols = size(dest)
    if n_rows != nrow(data) || n_cols != total_cols
        throw(DimensionMismatch("dest is $(size(dest)), but modelcols would yield $(nrow(data))×$total_cols"))
    end

    # Copy each block straight into dest
    col_start = 1
    for block in blocks
        if block isa AbstractVector
            @inbounds copyto!(view(dest, :, col_start), block)
            col_start += 1
        else
            nblock_cols = size(block, 2)
            @inbounds copyto!(view(dest, :, col_start:col_start+nblock_cols-1), block)
            col_start += nblock_cols
        end
    end

    return dest
end
