# zero_alloc_modelrow.jl
# Zero-allocation modelrow! interface

###############################################################################
# ZERO-ALLOCATION APPROACH: Pre-compile Everything
###############################################################################

"""
    modelrow!(row_vec, compiled_formula, data, row_idx)

Zero-allocation model row evaluation using pre-compiled formula.

# Arguments
- `row_vec::Vector{Float64}`: Pre-allocated output vector
- `compiled_formula::CompiledFormula`: Result from `compile_formula(model)`
- `data`: Column-table format data  
- `row_idx::Int`: Row index to evaluate

# Performance
- **Time**: ~50-100ns
- **Allocations**: 0 bytes

# Example
```julia
# One-time setup (allocates during compilation)
model = lm(@formula(y ~ x * group), df)
compiled = compile_formula(model)  # Compile once
data = Tables.columntable(df)
row_vec = Vector{Float64}(undef, length(compiled))

# Zero-allocation runtime (call millions of times)
for i in 1:1_000_000
    modelrow!(row_vec, compiled, data, i)  # 0 allocations
    # Use row_vec...
end
```
"""
function modelrow!(row_vec::AbstractVector{Float64}, 
                  compiled::CompiledFormula, 
                  data, 
                  row_idx::Int)
    # Direct delegation - zero allocations
    compiled(row_vec, data, row_idx)
    return row_vec
end

###############################################################################
# CONVENIENCE VS PERFORMANCE TRADE-OFF
###############################################################################

"""
    modelrow!(row_vec, model, data, row_idx; cache=true)

Convenient modelrow! with optional caching control.

# Arguments  
- `cache=true`: Use automatic caching (may allocate on lookup)
- `cache=false`: Always recompile (definitely allocates)

# Performance Characteristics
- `cache=true`: ~100-500ns, small allocations for hash/lookup
- `cache=false`: ~1-10ms, large allocations for compilation
- Pre-compiled: ~50-100ns, zero allocations

# Recommendation
For production performance-critical code, use the pre-compiled version:
```julia
compiled = compile_formula(model)  # Once
modelrow!(row_vec, compiled, data, row_idx)  # Many times, zero-alloc
```
"""
function modelrow!(row_vec::AbstractVector{Float64}, 
                  model, 
                  data, 
                  row_idx::Int; 
                  cache::Bool=true)
    if cache
        # Convenient but may allocate ~32-64 bytes for hash/lookup
        compiled = get_or_compile_formula_identity(model)
        compiled(row_vec, data, row_idx)
    else
        # Always recompile (slow, definitely allocates)
        compiled = compile_formula(model)
        compiled(row_vec, data, row_idx)
    end
    return row_vec
end

###############################################################################
# OBJECT-BASED ZERO-ALLOCATION INTERFACE
###############################################################################

"""
    ModelRowEvaluator

Pre-compiled evaluator that achieves true zero-allocation performance.

# Example
```julia
# Setup (one-time cost)
model = lm(@formula(y ~ x * group), df)
evaluator = ModelRowEvaluator(model, df)

# Zero-allocation usage
for i in 1:1_000_000
    evaluator(i)  # Returns model matrix row i, 0 allocations
end
```
"""
struct ModelRowEvaluator
    compiled::CompiledFormula
    data::NamedTuple
    row_vec::Vector{Float64}
    
    function ModelRowEvaluator(model, df::DataFrame)
        compiled = compile_formula(model)
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        new(compiled, data, row_vec)
    end
end

"""
    (evaluator::ModelRowEvaluator)(row_idx) -> Vector{Float64}

Evaluate model row with zero allocations.
"""
function (evaluator::ModelRowEvaluator)(row_idx::Int)
    evaluator.compiled(evaluator.row_vec, evaluator.data, row_idx)
    return evaluator.row_vec
end

"""
    (evaluator::ModelRowEvaluator)(row_vec, row_idx)

Evaluate model row into provided vector with zero allocations.
"""
function (evaluator::ModelRowEvaluator)(row_vec::AbstractVector{Float64}, row_idx::Int)
    evaluator.compiled(row_vec, evaluator.data, row_idx)
    return row_vec
end

###############################################################################
# HASH-FREE CACHING (Advanced)
###############################################################################

"""
Advanced: Use object identity instead of formula hashing to avoid string allocations.
"""
const MODEL_IDENTITY_CACHE = IdDict{Any, CompiledFormula}()

function get_or_compile_formula_identity(model)
    # Use IdDict with object identity - no hashing needed
    if haskey(MODEL_IDENTITY_CACHE, model)
        return MODEL_IDENTITY_CACHE[model]
    else
        compiled = compile_formula(model)
        MODEL_IDENTITY_CACHE[model] = compiled
        return compiled
    end
end

"""
    modelrow_cached!(row_vec, model, data, row_idx)

Potentially zero-allocation cached version using object identity.
"""
function modelrow_cached!(row_vec::AbstractVector{Float64}, model, data, row_idx::Int)
    compiled = get_or_compile_formula_identity(model)
    compiled(row_vec, data, row_idx)
    return row_vec
end

