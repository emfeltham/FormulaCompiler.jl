# types.jl - Core derivative types and data structures

"""
Dual cache entry for a specific dual type
"""
mutable struct DualCache{DualT, NTMerged, OverrideVecT}
    compiled_dual::Any  # UnifiedCompiled{DualT, Ops, S, O} - keeping Any for now due to type complexity
    rowvec_dual::Vector{DualT}
    data_over_dual::NTMerged
    overrides_dual::OverrideVecT
    x_dual::Vector{DualT}
    last_row::Int
end

"""
Cache container for dual types keyed by (Type, row)
"""
struct DualCacheDict
    caches::Dict{DataType, DualCache}
end

DualCacheDict() = DualCacheDict(Dict{DataType, DualCache}())

function get_or_create_cache!(cache_dict::DualCacheDict, ::Type{DualT}, de, row::Int) where {DualT}
    if haskey(cache_dict.caches, DualT)
        cache = cache_dict.caches[DualT]
        # Update row without rebuilding
        cache.last_row = row
        return cache
    else
        # Build new cache entry
        UB = typeof(de.compiled_base)
        OpsT = UB.parameters[2]
        ST = UB.parameters[3]
        OT = UB.parameters[4]
        
        compiled_dual = UnifiedCompiled{DualT, OpsT, ST, OT}(de.compiled_base.ops)
        rowvec_dual = Vector{DualT}(undef, length(de))
        data_over_dual, overrides_dual = build_row_override_data_typed(de.base_data, de.vars, row, DualT)
        x_dual = Vector{DualT}(undef, length(de.vars))
        
        cache = DualCache(compiled_dual, rowvec_dual, data_over_dual, overrides_dual, x_dual, row)
        cache_dict.caches[DualT] = cache
        return cache
    end
end

"""
Callable closure for ForwardDiff that writes into a reusable buffer
"""
struct DerivClosure{DE}
    de_ref::Base.RefValue{DE}
end

Base.length(g::DerivClosure) = length(g.de_ref[])

# More specific constructor for better type inference
DerivClosure(de::DE) where {DE} = DerivClosure{DE}(Base.RefValue{DE}(de))

"""
Scalar gradient closure for η = Xβ that reuses the vector closure
"""
struct GradClosure{GV}
    gvec::GV
    beta_ref::Base.RefValue{Vector{Float64}}
end

@inline function (gc::GradClosure)(x)
    v = gc.gvec(x)
    return dot(gc.beta_ref[], v)
end

function (g::DerivClosure)(x::AbstractVector)
    de = g.de_ref[]
    Tx = eltype(x)
    
    # Select compiled + buffer based on element type
    if Tx === Float64
        compiled_T = de.compiled_base
        row_vec = de.rowvec_float
        ov_vec = de.overrides
        data_over = de.data_over
    else
        # Use cached dual structures - no rebuilding per call
        cache = get_or_create_cache!(de.dual_cache, Tx, de, de.row)
        compiled_T = cache.compiled_dual
        row_vec = cache.rowvec_dual
        ov_vec = cache.overrides_dual
        data_over = cache.data_over_dual
    end
    
    # Update overrides for current row and x (in-place, no allocation)
    for i in eachindex(de.vars)
        ov = ov_vec[i]
        ov.row = de.row
        ov.replacement = x[i]
    end
    
    # Evaluate using appropriate merged data
    compiled_T(row_vec, data_over, de.row)
    return row_vec
end

"""
    DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC, JR, GR, NTDual, OVDualVecT}

Core derivative evaluator that maintains all state needed for zero-allocation 
derivative computations using both ForwardDiff and finite differences.

# Fields
- `compiled_base`: Base compiled formula evaluator
- `base_data`: Original column-table data  
- `vars`: Variables to differentiate with respect to
- `xbuf`: Buffer for variable values
- `overrides`: Concrete Float64 override vectors for FD
- `data_over`: Merged data with Float64 overrides
- `dual_cache`: Cache for all dual-typed structures (zero-allocation)
- `rowvec_float`: Buffer for Float64 results
- `g`: Concrete ForwardDiff closure
- `cfg`: Concrete ForwardDiff Jacobian configuration
- `gscalar`: Scalar gradient closure for η marginal effects
- `gradcfg`: Gradient configuration
- `beta_ref`: Reference to coefficient vector
- `row`: Current row being processed
- `jacobian_buffer`: Preallocated Jacobian matrix
- `eta_gradient_buffer`: Buffer for η gradients
- `xrow_buffer`: Buffer for model row evaluation
- `fd_yplus`, `fd_yminus`, `fd_xbase`: FD computation buffers
- `fd_columns`: Pre-cached column references for FD
"""
mutable struct DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC, JR, GR, NTDual, OVDualVecT}
    compiled_base::UnifiedCompiled{T, Ops, S, O}
    base_data::NTBase
    vars::Vector{Symbol}
    xbuf::Vector{Float64}
    # Prebuilt row-local overrides and merged data (Float64 path)
    overrides::Vector{FDOverrideVector}
    data_over::NTMerged
    # Dual cache system (replaces all Any dual fields)
    dual_cache::DualCacheDict
    # Typed single-cache for manual dual path (uses Tag=Nothing)
    compiled_dual_vec::UnifiedCompiled{ForwardDiff.Dual{Nothing, Float64, NV}, Ops, S, O}
    rowvec_dual_vec::Vector{ForwardDiff.Dual{Nothing, Float64, NV}}
    overrides_dual_vec::OVDualVecT
    data_over_dual_vec::NTDual
    x_dual_vec::Vector{ForwardDiff.Dual{Nothing, Float64, NV}}
    partials_unit_vec::Vector{ForwardDiff.Partials{NV, Float64}}
    # Row buffers
    rowvec_float::Vector{Float64}
    # AD vector closure and Jacobian config (concrete types)
    g::G
    cfg::JC
    # Scalar gradient closure and config for η (concrete types)
    gscalar::GS
    gradcfg::GC
    # Beta reference for scalar gradient path
    beta_ref::Base.RefValue{Vector{Float64}}
    # Internal beta buffer to avoid allocations when β is not Vector{Float64}
    beta_buf::Vector{Float64}
    # DiffResult containers for zero-allocation AD
    jac_result::JR
    grad_result::GR
    row::Int
    # Preallocated Jacobian matrix for marginal effects
    jacobian_buffer::Matrix{Float64}
    # Preallocated buffers for marginal effects mu
    eta_gradient_buffer::Vector{Float64}
    xrow_buffer::Vector{Float64}
    # Zero-allocation finite differences buffers
    fd_yplus::Vector{Float64}
    fd_yminus::Vector{Float64}
    fd_xbase::Vector{Float64}
    # Pre-cached column references for FD as NTuple (fully concrete, unrolled access)
    fd_columns::ColsT
end

Base.length(de::DerivativeEvaluator) = de.compiled_base |> length

# Note: For scenario compatibility with derivative evaluators, use the standalone
# derivative functions (e.g., derivative_modelrow_fd!) which accept arbitrary data
# rather than the cached evaluator versions that depend on specific base_data

