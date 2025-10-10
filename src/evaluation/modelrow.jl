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
    _column_type_category(col) -> Symbol or Type

Classify column by compilation behavior for cache key construction.

Returns semantic category symbols for types that compile identically,
concrete types for unknown/custom types that may have type-specific compilation.

# Semantic Categories
- `:numeric` - All numeric types (Int, Float64, etc.) compile to LoadOp
- `:bool` - Boolean vectors compile to LoadOp with special handling
- `:categorical` - CategoricalArray compiles to ContrastOp
- `:mixture` - CategoricalMixture compiles to MixtureContrastOp

# Cache Behavior
- Same category → cache HIT (e.g., Vector{Int} and Vector{Float64} both → :numeric)
- Different category → cache MISS (e.g., :categorical vs :mixture)

# Examples
```jldoctest
julia> _column_type_category(Vector{Float64}([1.0, 2.0]))
:numeric

julia> _column_type_category(Vector{Int}([1, 2]))
:numeric  # Same as Float64 - both compile to LoadOp

julia> _column_type_category(categorical(["A", "B"]))
:categorical

julia> _column_type_category(mix("A" => 0.5, "B" => 0.5))
:mixture  # Different from :categorical - compiles to MixtureContrastOp
```
"""
function _column_type_category(col)
    # Categorical mixtures MUST be distinguished from regular categoricals
    # (mixture → MixtureContrastOp, categorical → ContrastOp)
    col isa AbstractVector{<:CategoricalMixture} && return :mixture

    # Regular categorical variables
    col isa CategoricalVector && return :categorical

    # Boolean vectors (special handling in some contexts)
    col isa AbstractVector{Bool} && return :bool

    # All numeric types compile identically to LoadOp
    # (Int, Float64, Float32, etc. all behave the same)
    col isa AbstractVector{<:Real} && return :numeric

    # Unknown/custom types - use concrete type to be safe
    # (may have type-specific compilation behavior we don't know about)
    return typeof(col)
end

"""
    get_or_compile_formula(model, data)

Get cached compiled formula or compile new one with semantic type-aware caching.

# Cache Key Strategy
Creates cache key based on:
1. Model object (coefficients, structure)
2. Column names (formula structure)
3. Semantic type categories (compilation behavior)

# Type Category Benefits
- **Better cache hits**: Vector{Int} and Vector{Float64} share cache entry
- **Correct mixture handling**: CategoricalArray vs CategoricalMixture distinguished
- **Future-proof**: New types can be added to category system

# Examples
```julia
# These share a cache entry (both :numeric):
data1 = (x = Float64[1.0, 2.0], y = ...)
data2 = (x = Int[1, 2], y = ...)  # Cache HIT ✓

# These get separate entries (different compilation):
data3 = (edu = categorical(["HS"]), ...)      # :categorical
data4 = (edu = mix("HS" => 0.5, "C" => 0.5), ...)  # :mixture - Cache MISS ✓
```
"""
function get_or_compile_formula(model, data)
    # Semantic type signature based on compilation behavior
    type_sig = Tuple(_column_type_category(v) for (k, v) in pairs(data))

    # Comprehensive cache key: model + column names + semantic types
    cache_key = (model, hash((keys(data), type_sig)))

    # Get from cache or compile and store
    return get!(COMPILED_MODEL_CACHE, cache_key) do
        compile_formula(model, data)
    end
end

###############################################################################
# CORE MODELROW! FUNCTION
###############################################################################

"""
    modelrow!(output, compiled, data, row_idx) -> output

Evaluate a single model matrix row in-place with zero allocations.

The primary interface for high-performance row evaluation. This function provides
zero-allocation evaluation, making it suitable for tight computational loops and
performance-critical applications.

# Arguments
- `output::AbstractVector{Float64}`: Pre-allocated output vector (modified in-place)
  - Must have length ≥ `length(compiled)`
  - Contents will be overwritten with model matrix row values
- `compiled`: Compiled formula from `compile_formula(model, data)`
- `data`: Data in Tables.jl format (preferably `Tables.columntable(df)` for best performance)
- `row_idx::Int`: Row index to evaluate (1-based indexing)

# Returns
- `output`: The same vector passed in, now containing the evaluated model matrix row

# Performance
- **Memory**: Zero bytes allocated after warmup
- **Scaling**: Constant time regardless of dataset size or formula complexity
- **Validation**: Tested across 2000+ diverse formula configurations

# Example
```julia
using FormulaCompiler, GLM, Tables

# Setup (one-time cost)
model = lm(@formula(y ~ x * group + log(z)), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)
output = Vector{Float64}(undef, length(compiled))

# High-performance evaluation (repeated many times)
modelrow!(output, compiled, data, 1)    # Zero allocations
modelrow!(output, compiled, data, 100)  # Zero allocations

# Monte Carlo simulation example
for i in 1:1_000_000
    row_idx = rand(1:nrow(df))
    modelrow!(output, compiled, data, row_idx)  # Zero allocations each call
    # Process output...
end
```

# Error Handling
- `BoundsError`: If `row_idx` exceeds data size
- `DimensionMismatch`: If `output` vector is too small
- Validates arguments in debug builds

See also: [`modelrow`](@ref) for allocating version, [`compile_formula`](@ref), [`ModelRowEvaluator`](@ref)
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
    ModelRowEvaluator{T,Ops,S,O}

Object-oriented interface for reusable, pre-compiled model evaluation.

Combines compiled formula, data, and output buffer into a single object that can be
called repeatedly for both allocating and non-allocating row evaluation. Useful when
the same model and data will be evaluated many times.

# Type Parameters
- `T`: Element type (typically `Float64`)
- `Ops`: Compiled operations tuple type
- `S`: Scratch buffer size 
- `O`: Output vector size

# Fields
- `compiled::UnifiedCompiled`: Pre-compiled formula
- `data::NamedTuple`: Data in column-table format
- `row_vec::Vector{Float64}`: Internal buffer for non-allocating calls

# Constructors
```julia
ModelRowEvaluator(model, df::DataFrame)      # Converts DataFrame to column table
ModelRowEvaluator(model, data::NamedTuple)   # Uses data directly
```

# Interface
```julia
# Allocating interface - returns new vector
result = evaluator(row_idx)

# Non-allocating interface - uses provided vector  
evaluator(output_vector, row_idx)
```

# Performance
- **Construction**: One-time compilation cost
- **Allocating calls**: Fast evaluation plus allocation cost
- **Non-allocating calls**: Zero bytes allocated
- **Memory**: Minimal overhead beyond compiled formula and data reference

# Example
```julia
using FormulaCompiler, GLM

# Create evaluator (one-time setup)
model = lm(@formula(y ~ x * group + log(z)), df)
evaluator = ModelRowEvaluator(model, df)

# Allocating interface (convenient)
row_1 = evaluator(1)      # Returns Vector{Float64}
row_2 = evaluator(100)    # Returns Vector{Float64}

# Non-allocating interface (fast)
output = Vector{Float64}(undef, length(evaluator))
evaluator(output, 1)      # Zero allocations
evaluator(output, 100)    # Zero allocations

# Batch processing
results = Matrix{Float64}(undef, 1000, length(evaluator))
for i in 1:1000
    evaluator(view(results, i, :), i)  # Zero allocations
end
```

# When to Use
- **Repeated evaluation**: Same model and data used many times
- **Object-oriented style**: Prefer objects over function calls
- **Mixed interfaces**: Need both allocating and non-allocating evaluation
- **Clean encapsulation**: Bundle model, data, and buffer management

See also: [`modelrow!`](@ref), [`modelrow`](@ref), [`compile_formula`](@ref)
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
# POPULATION ANALYSIS NOTE
###############################################################################

# Population analysis should use simple loops with existing row-wise functions:
#
# # Example pattern for population effects:
# population_effects = Vector{Float64}(undef, n_rows)
# for row in 1:n_rows
#     # Use existing row-wise functions like marginal_effects_eta!
#     # with CounterfactualVector for single-row perturbations
#     population_effects[row] = compute_individual_effect(row)
# end
# population_ame = mean(population_effects)
#

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

Evaluate a single model matrix row, returning a new vector (allocating version).

Convenient interface for when pre-allocation is not practical. Uses internal
formula compilation and caching for performance optimization, though the
non-allocating `modelrow!` interface is preferred for performance-critical code.

# Arguments
- `model`: Fitted statistical model (GLM, MixedModel, etc.)
- `data`: Data in Tables.jl format 
- `row_idx::Int`: Row index to evaluate (1-based)

# Returns
- `Vector{Float64}`: New vector containing model matrix row values

# Performance
- **First call**: Includes one-time compilation cost
- **Subsequent calls**: Fast evaluation plus allocation cost for vector creation
- **Memory**: Allocates new vector each call
- **Caching**: Automatically caches compiled formula for reuse

# Example
```julia
using FormulaCompiler, GLM

model = lm(@formula(y ~ x * group + log(z)), df)
data = Tables.columntable(df)

# Convenient single-row evaluation
row_1 = modelrow(model, data, 1)      # First call (includes compilation)
row_2 = modelrow(model, data, 2)      # Subsequent calls (uses cached compilation)
row_100 = modelrow(model, data, 100)  # Fast (uses cached compilation)
```

# When to Use
- **Prototyping**: Quick analysis and exploration
- **Small datasets**: When allocation overhead is negligible
- **Convenience**: When code simplicity outweighs performance requirements

# Performance Alternative
For zero-allocation performance in loops, use [`modelrow!`](@ref):
```julia
output = Vector{Float64}(undef, length(compile_formula(model, data)))
for i in 1:n_iterations
    modelrow!(output, compiled, data, i)  # Zero allocations each iteration
end
```

See also: [`modelrow!`](@ref), [`ModelRowEvaluator`](@ref), [`compile_formula`](@ref)
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

