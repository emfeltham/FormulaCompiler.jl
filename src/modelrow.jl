# modelrow!.jl - Clean specialized-only implementation

###############################################################################
# CACHE FOR SPECIALIZED FORMULAS ONLY
###############################################################################

"""
Cache for specialized formulas using model+data structure as key.
"""
const SPECIALIZED_MODEL_CACHE = Dict{Any, Any}()

###############################################################################
# CORE MODELROW! FUNCTION - SPECIALIZED ONLY
###############################################################################

"""
    modelrow!(row_vec, specialized_formula, data, row_idx)

Model row evaluation using specialized formulas.
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    specialized::SpecializedFormula{D, O}, 
    data, 
    row_idx::Int
) where {D, O}
    @assert length(row_vec) >= length(specialized) "Vector too small: need $(length(specialized)), got $(length(row_vec))"
    @assert 1 <= row_idx <= length(first(data)) "Invalid row index: $row_idx (data has $(length(first(data))) rows)"

    specialized(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow!(row_vec, model, data, row_idx; cache=true)

Convenient modelrow! with automatic compilation to specialized formulas.

N.B., this allocates alot more when cache=false.
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    model::Union{LinearModel, GeneralizedLinearModel, LinearMixedModel, GeneralizedLinearMixedModel, StatsModels.TableRegressionModel}, 
    data, 
    row_idx::Int; 
    cache::Bool=true
)
    if cache
        specialized_formula = get_or_compile_specialized_formula(model, data)
    else
        specialized_formula = compile_formula_specialized(model, data)
    end
    
    specialized_formula(row_vec, data, row_idx)
    return row_vec
end

###############################################################################
# SPECIALIZED MODEL ROW EVALUATOR - CLEAN VERSION
###############################################################################

"""
    ModelRowEvaluator{D, O}

Pre-compiled evaluator using specialized formulas only.
"""
struct ModelRowEvaluator{D, O}
    specialized::SpecializedFormula{D, O}
    data::NamedTuple
    row_vec::Vector{Float64}
    
    function ModelRowEvaluator(model, df::DataFrame)
        data = Tables.columntable(df)
        specialized = compile_formula_specialized(model, data)
        row_vec = Vector{Float64}(undef, length(specialized))
        
        D = typeof(specialized.data)
        O = typeof(specialized.operations)
        
        new{D, O}(specialized, data, row_vec)
    end
end

"""
    (evaluator::ModelRowEvaluator{D, O})(row_idx) -> Vector{Float64}

Evaluate model row using specialized formula.
"""
function (evaluator::ModelRowEvaluator{D, O})(row_idx::Int) where {D, O}
    evaluator.specialized(evaluator.row_vec, evaluator.data, row_idx)
    return evaluator.row_vec
end

"""
    (evaluator::ModelRowEvaluator{D, O})(row_vec, row_idx)

Evaluate model row into provided vector.
"""
function (evaluator::ModelRowEvaluator{D, O})(row_vec::AbstractVector{Float64}, row_idx::Int) where {D, O}
    evaluator.specialized(row_vec, evaluator.data, row_idx)
    return row_vec
end

###############################################################################
# CACHING SYSTEM - SPECIALIZED ONLY
###############################################################################

"""
    get_or_compile_specialized_formula(model, data)

Get cached specialized formula or compile new one.
"""
function get_or_compile_specialized_formula(model, data)
    cache_key = (model, hash(keys(data)))
    
    if haskey(SPECIALIZED_MODEL_CACHE, cache_key)
        return SPECIALIZED_MODEL_CACHE[cache_key]
    else
        specialized = compile_formula_specialized(model, data)
        SPECIALIZED_MODEL_CACHE[cache_key] = specialized
        return specialized
    end
end

###############################################################################
# BATCH EVALUATION
###############################################################################

"""
    modelrow_batch!(matrix, specialized_formula, data, row_indices)

Batch evaluation using specialized formulas.
"""
function modelrow_batch!(
    matrix::AbstractMatrix{Float64}, 
    specialized::SpecializedFormula{D, O}, 
    data, 
    row_indices::AbstractVector{Int}
) where {D, O}
    @assert size(matrix, 1) >= length(row_indices) "Matrix height insufficient"
    @assert size(matrix, 2) == length(specialized) "Matrix width mismatch"
    
    @inbounds for (i, row_idx) in enumerate(row_indices)
        row_view = view(matrix, i, :)
        specialized(row_view, data, row_idx)
    end
    
    return matrix
end

"""
    modelrow_batch!(matrix, model, data, row_indices; cache=true)

Convenient batch evaluation.
"""
function modelrow_batch!(
    matrix::AbstractMatrix{Float64}, 
    model::Union{LinearModel, GeneralizedLinearModel, LinearMixedModel, GeneralizedLinearMixedModel, StatsModels.TableRegressionModel}, 
    data, 
    row_indices::AbstractVector{Int}; 
    cache::Bool=true
)
    if cache
        specialized_formula = get_or_compile_specialized_formula(model, data)
    else
        specialized_formula = compile_formula_specialized(model, data)
    end
    
    return modelrow_batch!(matrix, specialized_formula, data, row_indices)
end

###############################################################################
# CACHE MANAGEMENT
###############################################################################

"""
    clear_model_cache!()

Clear the specialized model cache.
"""
function clear_model_cache!()
    empty!(SPECIALIZED_MODEL_CACHE)
    println("Specialized model cache cleared.")
end

###############################################################################
# MODELROW - Non-mutating (allocating) versions using specialized formulas
###############################################################################

"""
    modelrow(model, data, row_idx) -> Vector{Float64}

Evaluate a single row and return a new vector (allocating version).
Uses specialized formulas for optimal performance.

# Example
```julia
row_values = modelrow(model, data, 1)  # Returns Vector{Float64}
```
"""
function modelrow(model, data, row_idx::Int)
    specialized = get_or_compile_specialized_formula(model, data)
    row_vec = Vector{Float64}(undef, length(specialized))
    specialized(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow(model, data, row_indices) -> Matrix{Float64}

Evaluate multiple rows and return a new matrix (allocating version).
Uses specialized formulas for optimal performance.

# Example
```julia
matrix = modelrow(model, data, [1, 5, 10])  # Returns Matrix{Float64}
```
"""
function modelrow(model, data, row_indices::AbstractVector{Int})
    specialized = get_or_compile_specialized_formula(model, data)
    matrix = Matrix{Float64}(undef, length(row_indices), length(specialized))
    modelrow_batch!(matrix, specialized, data, row_indices)
    return matrix
end

"""
    modelrow(specialized_formula, data, row_idx) -> Vector{Float64}

Evaluate a single row with pre-compiled specialized formula.

# Example
```julia
specialized = compile_formula_specialized(model, data)
row_values = modelrow(specialized, data, 1)  # Returns Vector{Float64}
```
"""
function modelrow(
    specialized::SpecializedFormula{D, O}, 
    data, 
    row_idx::Int
) where {D, O}
    row_vec = Vector{Float64}(undef, length(specialized))
    specialized(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow(specialized_formula, data, row_indices) -> Matrix{Float64}

Evaluate multiple rows with pre-compiled specialized formula.

# Example
```julia
specialized = compile_formula_specialized(model, data)
matrix = modelrow(specialized, data, [1, 5, 10])  # Returns Matrix{Float64}
```
"""
function modelrow(
    specialized::SpecializedFormula{D, O}, 
    data, 
    row_indices::AbstractVector{Int}
) where {D, O}
    matrix = Matrix{Float64}(undef, length(row_indices), length(specialized))
    modelrow_batch!(matrix, specialized, data, row_indices)
    return matrix
end
