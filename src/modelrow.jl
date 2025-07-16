# modelrow.jl
# Allocating modelrow interface with fixed method signatures

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

###############################################################################
# CONVENIENCE ALLOCATING VERSIONS
###############################################################################

"""
    modelrow(compiled_formula, data, row_indices) -> Matrix{Float64}

Allocating version that returns a new matrix.

# Example
```julia
matrix = modelrow(compiled, data, [1, 5, 10])  # Returns Matrix{Float64}
```
"""
function modelrow(
    compiled::CompiledFormula, 
    data, 
    row_indices::Vector{Int}
)
    matrix = Matrix{Float64}(undef, length(row_indices), length(compiled))
    modelrow!(matrix, compiled, data, row_indices)
    return matrix
end

"""
    modelrow(model, data, row_indices; cache=true) -> Matrix{Float64}

Convenient allocating version with automatic caching.

# Example
```julia
matrix = modelrow(model, data, [1, 5, 10])  # Returns Matrix{Float64}
```
"""
function modelrow(
    model::Union{LinearModel, GeneralizedLinearModel, LinearMixedModel, GeneralizedLinearMixedModel, StatsModels.TableRegressionModel}, 
    data, 
    row_indices::Vector{Int}; 
    cache::Bool=true
)
    if cache
        compiled = get_or_compile_formula_identity(model)
    else
        compiled = compile_formula(model)
    end
    
    matrix = Matrix{Float64}(undef, length(row_indices), length(compiled))
    modelrow!(matrix, compiled, data, row_indices)
    return matrix
end
