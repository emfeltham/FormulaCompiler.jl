# types.jl - Separate concrete derivative evaluator types

"""
Abstract base type for all derivative evaluators
"""
abstract type AbstractDerivativeEvaluator end

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


"""
    FDEvaluator{T, Ops, S, O, NTBase, NTMerged} <: AbstractDerivativeEvaluator

Finite differences derivative evaluator with only essential type parameters (6 total).

Provides efficient finite difference computation for derivatives without carrying
any AD infrastructure. Uses NumericCounterfactualVector{Float64} for type-stable
counterfactual operations.

# Type Parameters
- `T, Ops, S, O`: Required by FormulaCompiler's position mapping system
- `NTBase, NTMerged`: Ensure concrete NamedTuple types for type-stable data access

# Fields
- `compiled_base`: Base compiled formula evaluator
- `base_data`: Original column-table data
- `vars`: Variables to differentiate with respect to
- `counterfactuals`: Tuple of NumericCounterfactualVector{Float64} only
- `data_counterfactual`: Merged data with Float64 counterfactuals
- `y_plus`, `yminus`, `xbase`: FD computation buffers
- `jacobian_buffer`: Preallocated Jacobian matrix
- `xrow_buffer`: Buffer for model row evaluation
- `row`: Current row being processed

# Performance
- **Zero field pollution**: No unused AD fields
- **Memory efficient**: Only carries FD infrastructure
- **Type stable**: Concrete Float64 counterfactuals throughout
"""
mutable struct FDEvaluator{T, Ops, S, O, NTBase, NTMerged, CF} <: AbstractDerivativeEvaluator
    # Common fields
    compiled_base::UnifiedCompiled{T, Ops, S, O}
    base_data::NTBase
    vars::Vector{Symbol}

    # FD-specific counterfactual system
    counterfactuals::CF  # Tuple of NumericCounterfactualVector{Float64} only
    data_counterfactual::NTMerged

    # FD-only fields (no AD pollution)
    y_plus::Vector{Float64}
    yminus::Vector{Float64}
    xbase::Vector{Float64}
    jacobian_buffer::Matrix{Float64}
    xrow_buffer::Vector{Float64}
    row::Int
end

"""
    ADEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, G, JC} <: AbstractDerivativeEvaluator

Automatic differentiation evaluator with essential + AD-specific type parameters (9 total).

Provides ForwardDiff-based automatic differentiation without carrying any FD infrastructure.
Uses NumericCounterfactualVector{Dual{...}} for type-stable dual number operations.

# Type Parameters
- `T, Ops, S, O`: Required by FormulaCompiler's position mapping system
- `NTBase, NTMerged`: Ensure concrete NamedTuple types for type-stable data access
- `NV`: ForwardDiff dual dimensionality (ForwardDiff.Dual{Nothing, Float64, NV})
- `G, JC`: Concrete ForwardDiff closure/config types for zero-allocation AD

# Fields
- `compiled_base`: Base compiled formula evaluator (Float64)
- `compiled_dual`: Dual-specialized compiled evaluator (T = Dual type)
- `base_data`: Original column-table data
- `vars`: Variables to differentiate with respect to
- `counterfactuals`: Tuple of NumericCounterfactualVector{Dual{...}} only
- `data_counterfactual`: Merged data with Dual counterfactuals
- `x_dual_vec`, `partials_unit_vec`, `rowvec_dual_vec`: AD computation buffers
- `jacobian_buffer`: Preallocated Jacobian matrix
- `xrow_buffer`: Buffer for model row evaluation
- `g`, `cfg`: Concrete ForwardDiff closure and configuration
- `row`: Current row being processed

# Performance
- **Zero field pollution**: No unused FD fields
- **Memory efficient**: Only carries AD infrastructure
- **Type stable**: Concrete Dual counterfactuals throughout
- **Zero allocation**: After warmup, all AD operations are allocation-free
"""
mutable struct ADEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, G, JC, CF} <: AbstractDerivativeEvaluator
    # Common fields
    compiled_base::UnifiedCompiled{Float64, Ops, S, O}  # Base always Float64
    compiled_dual::UnifiedCompiled{T, Ops, S, O}        # T = Dual type
    base_data::NTBase
    vars::Vector{Symbol}

    # AD-specific counterfactual system
    counterfactuals::CF  # Tuple of NumericCounterfactualVector{Dual{...}} only
    data_counterfactual::NTMerged

    # AD-only fields (no FD pollution)
    x_dual_vec::Vector{T}
    partials_unit_vec::Vector{ForwardDiff.Partials{NV, Float64}}
    rowvec_dual_vec::Vector{T}
    jacobian_buffer::Matrix{Float64}
    xrow_buffer::Vector{Float64}

    # ForwardDiff configuration (concrete types for zero allocations)
    g::G
    cfg::JC
    row::Int

    # Beta handling infrastructure for marginal_effects_eta!
    beta_ref::Ref{Vector{Float64}}
    beta_buf::Vector{Float64}
end

# Union type for method dispatch compatibility
const derivativeevaluator = Union{FDEvaluator, ADEvaluator}

Base.length(de::AbstractDerivativeEvaluator) = length(de.compiled_base)

# Note: For scenario compatibility with derivative evaluators, use the standalone
# derivative functions (e.g., derivative_modelrow_fd!) which accept arbitrary data
# rather than the cached evaluator versions that depend on specific base_data

# Closure implementation for ADEvaluator (defined after type definitions to avoid forward reference)
function (g::DerivClosure{<:ADEvaluator})(x::AbstractVector)
    de = g.de_ref[]

    # Update counterfactuals for current row and x (in-place, no allocation)
    for i in eachindex(de.vars)
        cf = de.counterfactuals[i]
        update_counterfactual_row!(cf, de.row)
        update_counterfactual_replacement!(cf, x[i])
    end

    # Evaluate using dual-specialized compiled evaluator
    de.compiled_dual(de.rowvec_dual_vec, de.data_counterfactual, de.row)
    return de.rowvec_dual_vec
end

