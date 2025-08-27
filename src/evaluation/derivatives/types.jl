# types.jl - Core derivative types and data structures

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
    # Select compiled + buffer (initialize dual on first use)
    if Tx === Float64
        compiled_T = de.compiled_base
        row_vec = de.rowvec_float
    else
        # Ensure compiled_dual and rowvec_dual match current Dual element type (incl. Tag)
        UB = typeof(de.compiled_base)
        OpsT = UB.parameters[2]
        ST = UB.parameters[3]
        OT = UB.parameters[4]
        if (de.compiled_dual === nothing) || !(de.compiled_dual isa UnifiedCompiled{Tx, OpsT, ST, OT})
            de.compiled_dual = UnifiedCompiled{Tx, OpsT, ST, OT}(de.compiled_base.ops)
        end
        if (de.rowvec_dual === nothing) || (eltype(de.rowvec_dual) !== Tx) || (length(de.rowvec_dual) != length(de))
            de.rowvec_dual = Vector{Tx}(undef, length(de))
        end
        compiled_T = de.compiled_dual
        row_vec = de.rowvec_dual
    end
    # Select overrides and merged data based on element type
    if Tx === Float64
        ov_vec = de.overrides
        data_over = de.data_over
    else
        # For Dual types, we need to rebuild data_over for each new row
        # because categorical variables aren't wrapped and their row access is fixed
        need_build = de.overrides_dual === nothing
        if !need_build
            # Check if we have the right type AND the right row
            ovs = de.overrides_dual
            if !(isempty(ovs))
                stored_T = typeof(first(ovs)).parameters[1]
                stored_row = first(ovs).row
                need_build = (stored_T !== Tx) || (stored_row !== de.row)
            else
                need_build = true
            end
        end
        if need_build
            data_over_dual, overrides_dual = build_row_override_data_typed(de.base_data, de.vars, de.row, Tx)
            de.overrides_dual = overrides_dual
            de.data_over_dual = data_over_dual
        end
        ov_vec = de.overrides_dual
        data_over = de.data_over_dual
    end
    # Update overrides for current row and x
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
    DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}

Core derivative evaluator that maintains all state needed for zero-allocation 
derivative computations using both ForwardDiff and finite differences.

# Fields
- `compiled_base`: Base compiled formula evaluator
- `base_data`: Original column-table data  
- `vars`: Variables to differentiate with respect to
- `xbuf`: Buffer for variable values
- `overrides`: Concrete Float64 override vectors for FD
- `data_over`: Merged data with Float64 overrides
- `overrides_dual`: Dual-typed overrides (lazily initialized)
- `data_over_dual`: Merged data with Dual overrides  
- `rowvec_float`: Buffer for Float64 results
- `rowvec_dual`: Buffer for Dual results
- `compiled_dual`: Dual-typed compiled evaluator
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
mutable struct DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}
    compiled_base::UnifiedCompiled{T, Ops, S, O}
    base_data::NTBase
    vars::Vector{Symbol}
    xbuf::Vector{Float64}
    # Prebuilt row-local overrides and merged data (Float64 path)
    overrides::Vector{FDOverrideVector}
    data_over::NTMerged
    # Dual-typed overrides and merged data (lazily initialized per Dual tag)
    overrides_dual::Any
    data_over_dual::Any
    # Row buffers
    rowvec_float::Vector{Float64}
    rowvec_dual::Any
    # Compiled dual instance (per Dual tag)
    compiled_dual::Any
    # AD vector closure and Jacobian config (concrete types)
    g::G
    cfg::JC
    # Scalar gradient closure and config for η (concrete types)
    gscalar::GS
    gradcfg::GC
    # Beta reference for scalar gradient path
    beta_ref::Base.RefValue{Vector{Float64}}
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