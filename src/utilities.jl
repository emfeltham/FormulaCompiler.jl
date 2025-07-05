
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