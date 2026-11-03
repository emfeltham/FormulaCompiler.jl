# modelrow.jl
# includes override (data scenario) versions

###############################################################################
# CACHE FOR COMPILED FORMULAS
###############################################################################

# Bounded, least-recently-used cache of compiled formulas used by `get_or_compile_formula`
# (and through it `modelrow`, `modelrow!(…, model, …)` and Margins).
#
# Key: (model, column names, semantic column types); see `get_or_compile_formula`. The key
# holds the model strongly: GLM and StatsModels model types are immutable structs, which
# cannot be weak keys (`WeakKeyDict`) or carry finalizers, so release on garbage collection
# is not possible in general. Instead the number of entries is bounded: when it exceeds
# `compiled_cache_capacity()` the least recently used entries are dropped, so at most that
# many models are kept alive by the cache. `clear_compiled_cache!()` empties it.
#
# Every access to the Dict and the clock happens under `_COMPILED_CACHE_LOCK`, so concurrent
# calls cannot corrupt it. Compilation itself runs outside the lock.

mutable struct _CompiledCacheEntry
    compiled::Any
    last_used::UInt64
end

const COMPILED_MODEL_CACHE = Dict{Any, _CompiledCacheEntry}()
const _COMPILED_CACHE_LOCK = ReentrantLock()
const _DEFAULT_COMPILED_CACHE_CAPACITY = 16
const _COMPILED_CACHE_CAPACITY = Ref(_DEFAULT_COMPILED_CACHE_CAPACITY)
const _COMPILED_CACHE_TICK = Ref{UInt64}(0)

"""
    compiled_cache_capacity() -> Int

Maximum number of compiled formulas kept by [`get_or_compile_formula`](@ref) (default 16).
See [`set_compiled_cache_capacity!`](@ref).
"""
compiled_cache_capacity() = _COMPILED_CACHE_CAPACITY[]

"""
    set_compiled_cache_capacity!(n::Integer) -> Int

Set the maximum number of entries in the compiled-formula cache used by
[`get_or_compile_formula`](@ref) (and by `modelrow`, `modelrow!(row, model, data, i)`),
evicting least-recently-used entries if the cache is now over capacity. Returns the
previous capacity. `n = 0` disables caching: every call compiles afresh and nothing is
retained.

Each entry keeps its fitted model alive (and with it the model's data and model matrix),
so a large capacity combined with many large models (e.g. a bootstrap) retains memory until
the entries are evicted or [`clear_compiled_cache!`](@ref) is called.
"""
function set_compiled_cache_capacity!(n::Integer)
    n >= 0 || throw(ArgumentError("compiled cache capacity must be ≥ 0, got $n"))
    lock(_COMPILED_CACHE_LOCK)
    try
        old = _COMPILED_CACHE_CAPACITY[]
        _COMPILED_CACHE_CAPACITY[] = Int(n)
        _evict_compiled_over_capacity!()
        return old
    finally
        unlock(_COMPILED_CACHE_LOCK)
    end
end

"""
    clear_compiled_cache!() -> nothing

Empty the compiled-formula cache used by [`get_or_compile_formula`](@ref), releasing the
fitted models (and their data) it references. Compiled objects already handed out remain
valid. The cache is bounded ([`set_compiled_cache_capacity!`](@ref)), so this is only needed
to release memory sooner, e.g. between bootstrap replicates or after a large analysis.
Safe to call while other tasks use the cache.
"""
function clear_compiled_cache!()
    lock(_COMPILED_CACHE_LOCK)
    try
        empty!(COMPILED_MODEL_CACHE)
    finally
        unlock(_COMPILED_CACHE_LOCK)
    end
    return nothing
end

# Caller holds _COMPILED_CACHE_LOCK. O(entries) scan per eviction; the cache is small.
function _evict_compiled_over_capacity!()
    cap = _COMPILED_CACHE_CAPACITY[]
    while length(COMPILED_MODEL_CACHE) > cap
        victim = nothing
        oldest = typemax(UInt64)
        for (k, e) in COMPILED_MODEL_CACHE
            if e.last_used < oldest
                oldest = e.last_used
                victim = k
            end
        end
        delete!(COMPILED_MODEL_CACHE, victim)
    end
    return nothing
end

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
```julia
_column_type_category(Vector{Float64}([1.0, 2.0]))
# :numeric

_column_type_category(Vector{Int}([1, 2]))
# :numeric  # Same as Float64 - both compile to LoadOp

_column_type_category(categorical(["A", "B"]))
# :categorical

_column_type_category(mix("A" => 0.5, "B" => 0.5))
# :mixture  # Different from :categorical - compiles to MixtureContrastOp
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

Categorical pools and level orders are deliberately not part of the key: a compiled
`ContrastOp` resolves values against the model's own contrast levels at run time (see
`LevelIndex`), so one compiled object is correct for any data with these columns,
including subsets, reordered or rebuilt pools, and reference grids.

# Cache Size and Memory
The cache is bounded: it keeps at most [`compiled_cache_capacity()`](@ref) entries (default
16) and drops the least recently used entry beyond that. Each entry holds a reference to
its fitted model, so up to that many models (with their data) stay alive until evicted.
Change the bound with [`set_compiled_cache_capacity!`](@ref) (`0` disables caching) and
release everything with [`clear_compiled_cache!`](@ref).

# Thread Safety
The cache itself is protected by a lock, so concurrent `get_or_compile_formula` calls are
safe. The compiled object it returns is shared process-wide and is **not** thread-safe (it
owns a scratch buffer): do not evaluate it from several tasks at once; for threaded work
call `compile_formula` once per task.

# Type Category Benefits
- Better cache hits: Vector{Int} and Vector{Float64} share cache entry
- Correct mixture handling: CategoricalArray vs CategoricalMixture distinguished
- Future-proof: New types can be added to category system

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
    type_sig = map(_column_type_category, values(data))

    # Cache key: model + column names + semantic types (the full values, not a hash, so
    # distinct keys can never collide)
    cache_key = (model, keys(data), type_sig)

    lock(_COMPILED_CACHE_LOCK)
    try
        entry = get(COMPILED_MODEL_CACHE, cache_key, nothing)
        if entry !== nothing
            entry.last_used = (_COMPILED_CACHE_TICK[] += 1)
            return entry.compiled
        end
    finally
        unlock(_COMPILED_CACHE_LOCK)
    end

    # Miss: compile outside the lock so a slow compilation does not block other callers
    compiled = compile_formula(model, data)

    lock(_COMPILED_CACHE_LOCK)
    try
        _COMPILED_CACHE_CAPACITY[] == 0 && return compiled
        entry = get(COMPILED_MODEL_CACHE, cache_key, nothing)
        tick = (_COMPILED_CACHE_TICK[] += 1)
        if entry !== nothing
            # Another task compiled the same key meanwhile; share its object
            entry.last_used = tick
            return entry.compiled
        end
        COMPILED_MODEL_CACHE[cache_key] = _CompiledCacheEntry(compiled, tick)
        _evict_compiled_over_capacity!()
    finally
        unlock(_COMPILED_CACHE_LOCK)
    end
    return compiled
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
    model::StatsModels.RegressionModel, 
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
    model::StatsModels.RegressionModel, 
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

# Deprecated names kept for compatibility (never exported).

"""
    clear_model_cache!()

Deprecated alias for [`clear_compiled_cache!`](@ref).
"""
clear_model_cache!() = clear_compiled_cache!()

"""
    cache_size() -> Int

Current number of entries in the compiled-formula cache.
"""
function cache_size()
    lock(_COMPILED_CACHE_LOCK)
    try
        return length(COMPILED_MODEL_CACHE)
    finally
        unlock(_COMPILED_CACHE_LOCK)
    end
end

"""
    cache_info() -> NamedTuple

Entries and capacity of the compiled-formula cache: `(entries = …, capacity = …)`.
"""
function cache_info()
    lock(_COMPILED_CACHE_LOCK)
    try
        return (entries = length(COMPILED_MODEL_CACHE), capacity = _COMPILED_CACHE_CAPACITY[])
    finally
        unlock(_COMPILED_CACHE_LOCK)
    end
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

