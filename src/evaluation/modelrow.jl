# modelrow.jl
# includes override (data scenario) versions

###############################################################################
# CACHE FOR COMPILED FORMULAS
###############################################################################

"""
Cache for compiled formulas using model+data structure as key.
"""
const COMPILED_MODEL_CACHE = Dict{Any, Any}()

"""
    get_or_compile_formula(model, data)

Get cached compiled formula or compile new one.
"""
function get_or_compile_formula(model, data)
    # Simple cache key based on model and data structure
    cache_key = (model, hash(keys(data)))
    
    if haskey(COMPILED_MODEL_CACHE, cache_key)
        return COMPILED_MODEL_CACHE[cache_key]
    else
        compiled = compile_formula(model, data)
        COMPILED_MODEL_CACHE[cache_key] = compiled
        return compiled
    end
end

###############################################################################
# CORE MODELROW! FUNCTION
###############################################################################

"""
    modelrow!(row_vec, compiled_formula, data, row_idx)

Evaluate a single row of the model matrix in-place (zero-allocation).

# Arguments
- `row_vec::AbstractVector{Float64}`: Pre-allocated output vector (modified in-place)
- `compiled_formula`: Compiled formula from `compile_formula`
- `data`: Data in Tables.jl format (preferably from `Tables.columntable`)
- `row_idx::Int`: Row index to evaluate

# Returns
- `row_vec`: The same vector passed in, now containing the evaluated row

# Example
```julia
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))
modelrow!(row_vec, compiled, data, 1)  # Zero allocations
```
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    compiled::UnifiedCompiled{T, Ops, S, O}, 
    data, 
    row_idx::Int
) where {T, Ops, S, O}
    @assert length(row_vec) >= length(compiled) "Vector too small: need $(length(compiled)), got $(length(row_vec))"
    @assert 1 <= row_idx <= length(first(data)) "Invalid row index: $row_idx (data has $(length(first(data))) rows)"

    compiled(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow!(row_vec, model, data, row_idx; cache=true)

Evaluate a single row of the model matrix in-place with automatic compilation.

# Arguments
- `row_vec::AbstractVector{Float64}`: Pre-allocated output vector (modified in-place)
- `model`: Statistical model (GLM, MixedModel, etc.)
- `data`: Data in Tables.jl format
- `row_idx::Int`: Row index to evaluate
- `cache::Bool`: Whether to cache compiled formula (default: true)

# Returns
- `row_vec`: The same vector passed in, now containing the evaluated row

# Example
```julia
model = lm(@formula(y ~ x + group), df)
data = Tables.columntable(df)
row_vec = Vector{Float64}(undef, size(modelmatrix(model), 2))
modelrow!(row_vec, model, data, 1)
```

!!! note
    First call compiles the formula. Subsequent calls reuse cached version when `cache=true`.
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    model::Union{LinearModel, GeneralizedLinearModel, LinearMixedModel, GeneralizedLinearMixedModel, StatsModels.TableRegressionModel}, 
    data, 
    row_idx::Int; 
    cache::Bool=true
)
    if cache
        compiled_formula = get_or_compile_formula(model, data)
    else
        compiled_formula = compile_formula(model, data)
    end
    
    compiled_formula(row_vec, data, row_idx)
    return row_vec
end

###############################################################################
# SPECIALIZED MODEL ROW EVALUATOR
###############################################################################

"""
    ModelRowEvaluator{D, O}

Pre-compiled evaluator using compiled formulas only.
"""
struct ModelRowEvaluator{T, Ops, S, O}
    compiled::UnifiedCompiled{T, Ops, S, O}
    data::NamedTuple
    row_vec::Vector{Float64}
    
    function ModelRowEvaluator(model, df::DataFrame)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        T = typeof(compiled).parameters[1]
        Ops = typeof(compiled.ops)
        S = typeof(compiled).parameters[3]  # ScratchSize
        O = typeof(compiled).parameters[4]  # OutputSize
        
        new{T, Ops, S, O}(compiled, data, row_vec)
    end
    
    function ModelRowEvaluator(model, data::NamedTuple)
        compiled = compile_formula(model, data)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        T = typeof(compiled).parameters[1]
        Ops = typeof(compiled.ops)
        S = typeof(compiled).parameters[3]
        O = typeof(compiled).parameters[4]
        
        new{T, Ops, S, O}(compiled, data, row_vec)
    end
end

"""
    (evaluator::ModelRowEvaluator{D, O})(row_idx) -> Vector{Float64}

Evaluate model row using compiled formula.
"""
function (evaluator::ModelRowEvaluator{T, Ops, S, O})(row_idx::Int) where {T, Ops, S, O}
    evaluator.compiled(evaluator.row_vec, evaluator.data, row_idx)
    return evaluator.row_vec
end

"""
    (evaluator::ModelRowEvaluator{D, O})(row_vec, row_idx)

Evaluate model row into provided vector.
"""
function (evaluator::ModelRowEvaluator{T, Ops, S, O})(row_vec::AbstractVector{Float64}, row_idx::Int) where {T, Ops, S, O}
    evaluator.compiled(row_vec, evaluator.data, row_idx)
    return row_vec
end

###############################################################################
# BATCH EVALUATION
###############################################################################

"""
    modelrow_batch!(matrix, compiled_formula, data, row_indices)

Batch evaluation using compiled formulas.
"""
function modelrow_batch!(
    matrix::AbstractMatrix{Float64}, 
    compiled::UnifiedCompiled{T, Ops, S, O}, 
    data, 
    row_indices::AbstractVector{Int}
) where {T, Ops, S, O}
    @assert size(matrix, 1) >= length(row_indices) "Matrix height insufficient"
    @assert size(matrix, 2) == length(compiled) "Matrix width mismatch"
    
    @inbounds for (i, row_idx) in enumerate(row_indices)
        row_view = view(matrix, i, :)
        compiled(row_view, data, row_idx)
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
        compiled_formula = get_or_compile_formula(model, data)
    else
        compiled_formula = compile_formula(model, data)
    end
    
    return modelrow_batch!(matrix, compiled_formula, data, row_indices)
end

###############################################################################
# SCENARIO-AWARE BATCH EVALUATION
###############################################################################

"""
    modelrow_scenarios!(matrix, model, scenarios, row_idx; cache=true)

Evaluate model row across multiple scenarios.
Each scenario gets its own properly cached compiled formula.
"""
function modelrow_scenarios!(
    matrix::AbstractMatrix{Float64},
    model::Union{LinearModel, GeneralizedLinearModel, LinearMixedModel, GeneralizedLinearMixedModel, StatsModels.TableRegressionModel},
    scenarios::Vector{DataScenario},
    row_idx::Int;
    cache::Bool=true
)
    n_scenarios = length(scenarios)
    
    # Get first scenario to determine output width
    first_compiled = if cache
        get_or_compile_formula(model, scenarios[1].data)
    else
        compile_formula(model, scenarios[1].data)
    end
    
    output_width = length(first_compiled)
    
    @assert size(matrix, 1) >= n_scenarios "Matrix height insufficient for scenarios"
    @assert size(matrix, 2) >= output_width "Matrix width insufficient for formula output"
    
    # Evaluate first scenario (already compiled)
    row_view = view(matrix, 1, 1:output_width)
    first_compiled(row_view, scenarios[1].data, row_idx)
    
    # Evaluate remaining scenarios
    for (i, scenario) in enumerate(scenarios[2:end])
        compiled = if cache
            get_or_compile_formula(model, scenario.data)
        else
            compile_formula(model, scenario.data)
        end
        
        row_view = view(matrix, i+1, 1:output_width)
        compiled(row_view, scenario.data, row_idx)
    end
    
    return matrix
end

"""
    modelrow_scenarios!(matrix, compiled::UnifiedCompiled, scenarios, row_idx)

Evaluate compiled formula across multiple scenarios.
Note: Same compiled formula used for all scenarios - may not be appropriate if formula 
depends on data structure.
"""
function modelrow_scenarios!(
    matrix::AbstractMatrix{Float64},
    compiled::UnifiedCompiled{Ops, S, O},
    scenarios::Vector{DataScenario},
    row_idx::Int
) where {Ops, S, O}
    @assert size(matrix, 1) >= length(scenarios) "Matrix height insufficient for scenarios"
    @assert size(matrix, 2) == length(compiled) "Matrix width mismatch"
    
    for (i, scenario) in enumerate(scenarios)
        row_view = view(matrix, i, :)
        compiled(row_view, scenario.data, row_idx)
    end
    
    return matrix
end

###############################################################################
# CACHE MANAGEMENT
###############################################################################

"""
    clear_model_cache!()

Clear the compiled model cache.
"""
function clear_model_cache!()
    empty!(COMPILED_MODEL_CACHE)
    println("Compiled model cache cleared.")
end

"""
    cache_size()

Get the current size of the model cache.
"""
function cache_size()
    return length(COMPILED_MODEL_CACHE)
end

"""
    cache_info()

Get detailed information about the cache contents.
"""
function cache_info()
    println("Model Cache Information")
    println("=" ^ 40)
    println("Total entries: $(length(COMPILED_MODEL_CACHE))")
    
    # Analyze cache keys
    has_overrides = 0
    no_overrides = 0
    
    for (key, _) in COMPILED_MODEL_CACHE
        if length(key) == 3  # Has override info
            has_overrides += 1
        else
            no_overrides += 1
        end
    end
    
    println("Entries without overrides: $no_overrides")
    println("Entries with overrides: $has_overrides")
    
    return (total=length(COMPILED_MODEL_CACHE), 
            with_overrides=has_overrides, 
            without_overrides=no_overrides)
end

###############################################################################
# MODELROW - Non-mutating (allocating) versions using compiled formulas
###############################################################################

"""
    modelrow(model, data, row_idx) -> Vector{Float64}

Evaluate a single row and return a new vector (allocating version).
Uses compiled formulas for optimal performance.

# Example
```julia
row_values = modelrow(model, data, 1)  # Returns Vector{Float64}
```
"""
function modelrow(model, data, row_idx::Int)
    compiled = get_or_compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow(model, data, row_indices) -> Matrix{Float64}

Evaluate multiple rows and return a new matrix (allocating version).
Uses compiled formulas for optimal performance.

# Example
```julia
matrix = modelrow(model, data, [1, 5, 10])  # Returns Matrix{Float64}
```
"""
function modelrow(model, data, row_indices::AbstractVector{Int})
    compiled = get_or_compile_formula(model, data)
    matrix = Matrix{Float64}(undef, length(row_indices), length(compiled))
    modelrow_batch!(matrix, compiled, data, row_indices)
    return matrix
end

"""
    modelrow(compiled_formula, data, row_idx) -> Vector{Float64}

Evaluate a single row with pre-compiled compiled formula.

# Example
```julia
compiled = compile_formula(model, data)
row_values = modelrow(compiled, data, 1)  # Returns Vector{Float64}
```
"""
function modelrow(
    compiled::UnifiedCompiled{T, Ops, S, O}, 
    data, 
    row_idx::Int
) where {T, Ops, S, O}
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow(compiled_formula, data, row_indices) -> Matrix{Float64}

Evaluate multiple rows with pre-compiled compiled formula.

# Example
```julia
compiled = compile_formula(model, data)
matrix = modelrow(compiled, data, [1, 5, 10])  # Returns Matrix{Float64}
```
"""
function modelrow(
    compiled::UnifiedCompiled{T, Ops, S, O}, 
    data, 
    row_indices::AbstractVector{Int}
) where {T, Ops, S, O}
    matrix = Matrix{Float64}(undef, length(row_indices), length(compiled))
    modelrow_batch!(matrix, compiled, data, row_indices)
    return matrix
end

"""
    modelrow(model, scenario::DataScenario, row_idx) -> Vector{Float64}

Evaluate model row using a data scenario (allocating version).
"""
function modelrow(model, scenario::DataScenario, row_idx::Int)
    compiled = get_or_compile_formula(model, scenario.data)
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, scenario.data, row_idx)
    return row_vec
end

"""
    modelrow(compiled::UnifiedCompiled, scenario::DataScenario, row_idx) -> Vector{Float64}

Evaluate model row using a data scenario with UnifiedCompiled (allocating version).
"""
function modelrow(compiled::UnifiedCompiled{T, Ops, S, O}, scenario::DataScenario, row_idx::Int) where {T, Ops, S, O}
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, scenario.data, row_idx)
    return row_vec
end
