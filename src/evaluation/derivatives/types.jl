# types.jl - Separate concrete derivative evaluator types

"""
    AbstractDerivativeEvaluator

Supertype of the evaluators built by [`derivativeevaluator`](@ref) (`:ad` and `:fd`
backends). Use it in signatures and field types that accept either backend. The public
interface of an evaluator `de` is:

- `length(de)`: the number of model-matrix columns;
- [`derivative_variables`](@ref)`(de)`: the variables it differentiates with respect to;
- [`derivative_backend`](@ref)`(de)`: `:ad` or `:fd`;
- [`derivative_modelrow!`](@ref) and [`derivative_modelrow`](@ref).

The concrete types and their fields are internal.
"""
abstract type AbstractDerivativeEvaluator end

# Tag type for FC's dual numbers. It is a `ForwardDiff.Tag`, so ForwardDiff's tag ordering
# (`≺`, used when duals with different tags meet in nested differentiation) is defined for it.
const FC_AD_TAG = ForwardDiff.Tag{Tuple{Val{:fc_ad}}, Float64}

# Chunk-size threshold: ForwardDiff's default (`DEFAULT_CHUNK_THRESHOLD = 12`). FC uses its
# own constant so the chunk size, and therefore the Jacobians, do not depend on ForwardDiff
# internals or on a `default_chunk_threshold` preference set for ForwardDiff.
const FC_AD_CHUNK_THRESHOLD = 12

"""
    _ad_chunksize(nvars, threshold = FC_AD_CHUNK_THRESHOLD) -> Int

Number of partials carried by each dual number for `nvars` input variables. Reproduces
ForwardDiff's chunk-size rule (as of ForwardDiff 0.10–1.4): `nvars` itself when
`nvars ≤ threshold` (vector mode, one pass); otherwise the smallest number of passes,
`ceil(nvars / threshold)`, with the variables spread evenly over them
(`ceil(nvars / npasses)` per pass).
"""
function _ad_chunksize(nvars::Integer, threshold::Integer = FC_AD_CHUNK_THRESHOLD)
    nvars <= threshold && return Int(nvars)
    nchunks = cld(nvars, threshold)
    return cld(nvars, nchunks)
end

"""
    _ad_seeds(Partials{N,V}) -> NTuple{N, Partials{N,V}}

The `N` unit seeds `e₁, …, e_N` (one in position `i`, zero elsewhere).
"""
@generated function _ad_seeds(::Type{ForwardDiff.Partials{N,V}}) where {N,V}
    seeds = [:(ForwardDiff.Partials{$N,V}(($([i == j ? :(one(V)) : :(zero(V)) for j in 1:N]...),)))
             for i in 1:N]
    return Expr(:tuple, seeds...)
end

"""
    FCJacobianConfig{T,V,N}

FC-owned work buffers for forward-mode Jacobians, replacing `ForwardDiff.JacobianConfig`
(whose fields and type parameters are not public API).

- `duals`: input buffer of `Dual{T,V,N}`, one per variable, reseeded for every chunk
- `seeds`: the `N` unit partials used to seed one chunk

Built only from `ForwardDiff.Dual` and `ForwardDiff.Partials`.
"""
struct FCJacobianConfig{T,V,N}
    duals::Vector{ForwardDiff.Dual{T,V,N}}
    seeds::NTuple{N,ForwardDiff.Partials{N,V}}
end

function FCJacobianConfig(::Type{T}, ::Type{V}, ::Val{N}, nvars::Int) where {T,V,N}
    duals = Vector{ForwardDiff.Dual{T,V,N}}(undef, nvars)
    return FCJacobianConfig{T,V,N}(duals, _ad_seeds(ForwardDiff.Partials{N,V}))
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
    JacobianContext{G, JC}

Container for ForwardDiff Jacobian computation infrastructure.
Breaks circular type dependency by isolating closure/config from main evaluator.

# Type Parameters
- `G`: Concrete DerivClosure type
- `JC`: Concrete `FCJacobianConfig` type

# Fields
- `g`: Callable closure evaluated on dual numbers by `fc_jacobian!`
- `cfg`: `FCJacobianConfig` (dual input buffer and seeds) for zero-allocation AD
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

"""
    derivative_variables(de::AbstractDerivativeEvaluator) -> Vector{Symbol}

The variables `de` differentiates with respect to, in the order of the Jacobian's columns:
column `j` of `derivative_modelrow!(J, de, row)` is ∂X/∂`derivative_variables(de)[j]`.

Returns the evaluator's own vector, without copying. Do not modify it.
"""
derivative_variables(de::AbstractDerivativeEvaluator) = de.vars

"""
    derivative_backend(de::AbstractDerivativeEvaluator) -> Symbol

`:ad` (automatic differentiation) or `:fd` (finite differences): the `backend` argument of
the [`derivativeevaluator`](@ref) call that built `de`. Pass it to `derivativeevaluator` to
build another evaluator with the same backend.
"""
derivative_backend(::ADEvaluator) = :ad
derivative_backend(::FDEvaluator) = :fd

# Note: For scenario compatibility with derivative evaluators, use the standalone
# derivative functions (e.g., derivative_modelrow_fd!) which accept arbitrary data
# rather than the cached evaluator versions that depend on specific base_data

# Closure implementation for ADEvaluator (defined after type definitions to avoid forward reference)
# Legacy closure - returns Vector{Dual} (allocating)
# In-place closure evaluated by fc_jacobian! (zero-allocation)
# Note: fc_jacobian! passes x as a vector of FC-tagged duals; y must be dual-typed too
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
