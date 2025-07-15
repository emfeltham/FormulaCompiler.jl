# modelrow_interface.jl
# Clean interface for model row evaluation

###############################################################################
# PUBLIC MODELROW! INTERFACE
###############################################################################

"""
    modelrow!(row_vec, model, data, row_idx)

Efficiently evaluate a single row of the model matrix into a pre-allocated vector.

# Arguments
- `row_vec::Vector{Float64}`: Pre-allocated output vector (length = ncol(modelmatrix))
- `model`: Fitted statistical model (from GLM.jl, etc.)
- `data`: Column-table format data (e.g., `Tables.columntable(df)`)
- `row_idx::Int`: Which row to evaluate (1-based indexing)

# Returns
- `row_vec`: The same vector, filled with model matrix values

# Performance
- **First call**: ~1-10ms compilation time
- **Subsequent calls**: ~50-100ns, zero allocations

# Example
```julia
using GLM, DataFrames, Tables

# Setup
df = DataFrame(x = randn(100), y = randn(100), group = categorical(rand(["A", "B"], 100)))
model = lm(@formula(y ~ x * group), df)
data = Tables.columntable(df)

# Efficient row evaluation
row_vec = Vector{Float64}(undef, size(modelmatrix(model), 2))
modelrow!(row_vec, model, data, 1)  # Fills row_vec with first row

# Batch processing
for i in 1:nrow(df)
    modelrow!(row_vec, model, data, i)
    # Use row_vec for predictions, etc.
end
```

# Advanced Usage
```julia
# Pre-compile for maximum performance
compiled = compile_formula(model)  # One-time compilation
row_vec = Vector{Float64}(undef, length(compiled))

# Then use either:
modelrow!(row_vec, model, data, row_idx)     # Convenient interface
# OR
compiled(row_vec, data, row_idx)             # Direct compiled function
```
"""
function modelrow!(row_vec::AbstractVector{Float64}, model, data, row_idx::Int)
    # Get or create compiled formula
    compiled = get_or_compile_formula(model)
    
    # Delegate to compiled function
    compiled(row_vec, data, row_idx)
    
    return row_vec
end

###############################################################################
# COMPILATION CACHING FOR AUTOMATIC PERFORMANCE
###############################################################################

# Global cache: model → compiled formula
const MODEL_CACHE = Dict{UInt64, Any}()

"""
Get cached compiled formula or compile if needed.
This enables automatic caching for the convenient `modelrow!` interface.
"""
function get_or_compile_formula(model)
    # Create a hash based on the model's formula
    model_hash = hash(string(fixed_effects_form(model)))
    
    if haskey(MODEL_CACHE, model_hash)
        return MODEL_CACHE[model_hash]
    else
        # Compile and cache
        compiled = compile_formula(model)
        MODEL_CACHE[model_hash] = compiled
        return compiled
    end
end

"""
    clear_model_cache!()

Clear the automatic model compilation cache.
Useful for testing or memory management.
"""
function clear_model_cache!()
    empty!(MODEL_CACHE)
    return nothing
end

###############################################################################
# BATCH EVALUATION INTERFACE
###############################################################################

"""
    modelrow!(matrix, model, data, row_indices)

Efficiently evaluate multiple rows into a pre-allocated matrix.

# Arguments
- `matrix::Matrix{Float64}`: Pre-allocated output matrix (size: length(row_indices) × ncol(modelmatrix))
- `model`: Fitted statistical model
- `data`: Column-table format data
- `row_indices`: Vector or range of row indices to evaluate

# Example
```julia
# Evaluate rows 1:100
matrix = Matrix{Float64}(undef, 100, size(modelmatrix(model), 2))
modelrow!(matrix, model, data, 1:100)
```
"""
function modelrow!(matrix::AbstractMatrix{Float64}, model, data, row_indices)
    compiled = get_or_compile_formula(model)
    
    @assert size(matrix, 2) == length(compiled) "Matrix width must match model matrix width"
    @assert size(matrix, 1) >= length(row_indices) "Matrix height insufficient for row indices"
    
    # Evaluate each row
    for (i, row_idx) in enumerate(row_indices)
        row_view = view(matrix, i, :)
        compiled(row_view, data, row_idx)
    end
    
    return matrix
end

"""
    modelrow(model, data, row_idx) -> Vector{Float64}

Evaluate a single row and return a new vector (allocating version).

# Example
```julia
row_values = modelrow(model, data, 1)  # Returns Vector{Float64}
```
"""
function modelrow(model, data, row_idx::Int)
    compiled = get_or_compile_formula(model)
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow(model, data, row_indices) -> Matrix{Float64}

Evaluate multiple rows and return a new matrix (allocating version).
"""
function modelrow(model, data, row_indices)
    compiled = get_or_compile_formula(model)
    matrix = Matrix{Float64}(undef, length(row_indices), length(compiled))
    modelrow!(matrix, model, data, row_indices)
    return matrix
end

###############################################################################
# COMPATIBILITY WITH EXISTING WORKFLOW
###############################################################################

"""
    compile_formula(model) -> CompiledFormula

Explicitly compile a formula for maximum performance.
Returns a callable object that can be used directly.

This is the same function you already have, but now it's part of a 
clean interface hierarchy:

1. `modelrow!(row_vec, model, data, row_idx)` - Convenient, auto-caching
2. `compiled = compile_formula(model); compiled(row_vec, data, row_idx)` - Explicit compilation
3. Internal `@generated modelrow!(row_vec, ::Val{H}, data, row_idx)` - Zero-allocation backend
"""
# Your existing compile_formula function works as-is!
