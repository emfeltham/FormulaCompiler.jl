# modelrow.jl

"""
    modelrow(model, data, row_idx) -> Vector{Float64}

Evaluate a single row and return a new vector (allocating version).

# Example
```julia
row_values = modelrow(model, data, 1)  # Returns Vector{Float64}
```
"""
function modelrow(model, data, row_idx::Int)
    compiled = get_or_compile_formula_identity(model)
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow(model, data, row_indices) -> Matrix{Float64}

Evaluate multiple rows and return a new matrix (allocating version).
"""
function modelrow(model, data, row_indices)
    compiled = get_or_compile_formula_identity(model)
    matrix = Matrix{Float64}(undef, length(row_indices), length(compiled))
    modelrow!(matrix, model, data, row_indices)
    return matrix
end
