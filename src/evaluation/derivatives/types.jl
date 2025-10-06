# types.jl - Separate concrete derivative evaluator types

"""
Abstract base type for all derivative evaluators
"""
abstract type AbstractDerivativeEvaluator end

const FC_AD_TAG = ForwardDiff.Tag{Tuple{Val{:fc_ad}}, Float64}

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
    JacobianContext{G, JC}

Container for ForwardDiff Jacobian computation infrastructure.
Breaks circular type dependency by isolating closure/config from main evaluator.

# Type Parameters
- `G`: Concrete DerivClosure type
- `JC`: Concrete JacobianConfig type

# Fields
- `g`: Callable closure for ForwardDiff.jacobian!
- `cfg`: Cached JacobianConfig for zero-allocation AD
- `input_vec`: Preallocated input buffer for row values
"""
struct JacobianContext{G, JC, VC}
    g::G
    cfg::JC
    input_vec::Vector{Float64}
    var_columns::VC
end

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
- `dual_output`: Scratch Dual vector for ForwardDiff.jacobian! primals
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
    ADEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, CF} <: AbstractDerivativeEvaluator

Automatic differentiation evaluator with essential + AD-specific type parameters (8 total).

Provides ForwardDiff-based automatic differentiation without carrying any FD infrastructure.
Uses NumericCounterfactualVector{Dual{...}} for type-stable dual number operations.

# Type Parameters
- `T, Ops, S, O`: Required by FormulaCompiler's position mapping system
- `NTBase, NTMerged`: Ensure concrete NamedTuple types for type-stable data access
- `NV`: ForwardDiff dual dimensionality (ForwardDiff.Dual{...})
- `CF`: Counterfactual tuple type

# Fields
- `compiled_base`: Base compiled formula evaluator (Float64)
- `compiled_dual`: Dual-specialized compiled evaluator (T = Dual type)
- `base_data`: Original column-table data
- `vars`: Variables to differentiate with respect to
- `counterfactuals`: Tuple of NumericCounterfactualVector{Dual{...}} only
- `data_counterfactual`: Merged data with Dual counterfactuals
- `jacobian_buffer`: Preallocated Jacobian matrix
- `xrow_buffer`: Buffer for model row evaluation
- `row`: Current row being processed

# Performance
- **Zero field pollution**: No unused FD fields
- **Memory efficient**: Only carries AD infrastructure
- **Type stable**: Concrete Dual counterfactuals throughout
- **Zero allocation**: After warmup, all AD operations are allocation-free
"""
mutable struct ADEvaluatorCore{T, Ops, S, O, NTBase, NTMerged, NV, CF}
    # Common fields
    compiled_base::UnifiedCompiled{Float64, Ops, S, O}  # Base always Float64
    compiled_dual::UnifiedCompiled{T, Ops, S, O}        # T = Dual type
    base_data::NTBase
    vars::Vector{Symbol}

    # AD-specific counterfactual system
    counterfactuals::CF  # Tuple of NumericCounterfactualVector{Dual{...}} only
    data_counterfactual::NTMerged

    dual_output::Vector{T}
    jacobian_buffer::Matrix{Float64}
    xrow_buffer::Vector{Float64}
    row::Int

    # Beta handling infrastructure for marginal_effects_eta!
    beta_ref::Ref{Vector{Float64}}
    beta_buf::Vector{Float64}
end

# Wrapper that couples an ADEvaluatorCore with its ForwardDiff context
struct ADEvaluator{Core<:ADEvaluatorCore, CTX} <: AbstractDerivativeEvaluator
    core::Core
    ctx::CTX
end

Base.getproperty(bundle::ADEvaluator, s::Symbol) = s === :ctx ? getfield(bundle, :ctx) : s === :core ? getfield(bundle, :core) : getproperty(getfield(bundle, :core), s)
Base.length(bundle::ADEvaluator) = length(getfield(bundle, :core))
Base.length(core::ADEvaluatorCore) = length(core.compiled_base)

@inline set_row!(core::ADEvaluatorCore, row::Int) = (core.row = row)
@inline current_row(core::ADEvaluatorCore) = core.row

# Union type for method dispatch compatibility
const derivativeevaluator = Union{FDEvaluator, ADEvaluator}

Base.length(de::AbstractDerivativeEvaluator) = length(de.compiled_base)

# Note: For scenario compatibility with derivative evaluators, use the standalone
# derivative functions (e.g., derivative_modelrow_fd!) which accept arbitrary data
# rather than the cached evaluator versions that depend on specific base_data

# Closure implementation for ADEvaluator (defined after type definitions to avoid forward reference)
# Legacy closure - returns Vector{Dual} (allocating)
# Phase 2: In-place closure for ForwardDiff.jacobian! (zero-allocation)
# Note: ForwardDiff.jacobian! expects f!(y, x) where it controls the types of y and x
# It will pass in Dual-typed x and expect Dual-typed y to be written
function (g::DerivClosure{<:ADEvaluatorCore})(y::AbstractVector, x::AbstractVector)
    de = g.de_ref[]

    # Update counterfactuals with values from x (ForwardDiff provides Dual values)
    row = current_row(de)
    @inbounds for i in eachindex(de.vars)
        cf = de.counterfactuals[i]
        update_counterfactual_row!(cf, row)
        update_counterfactual_replacement!(cf, x[i])
    end

    # Evaluate using dual-specialized compiled evaluator (writes to y, which ForwardDiff expects as Dual-typed)
    de.compiled_dual(y, de.data_counterfactual, row)
    return nothing  # ForwardDiff doesn't use return value for f! form
end
function (g::DerivClosure{<:ADEvaluatorCore})(x::AbstractVector)
    de = g.de_ref[]
    y = de.dual_output
    g(y, x)
    return y
end
