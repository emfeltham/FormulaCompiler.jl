# derivatives.jl

# Single-row override vector: returns replacement at `row`, base elsewhere (unused)
mutable struct SingleRowOverrideVector <: AbstractVector{Any}
    base::Any
    row::Int
    replacement::Any
end

########################### Marginal Effects (η, μ) ##########################

Base.size(v::SingleRowOverrideVector) = size(v.base)
Base.length(v::SingleRowOverrideVector) = length(v.base)
Base.IndexStyle(::Type{<:SingleRowOverrideVector}) = IndexLinear()
Base.eltype(::Type{SingleRowOverrideVector}) = Any
Base.getindex(v::SingleRowOverrideVector, i::Int) = (i == v.row ? v.replacement : getindex(v.base, i))

# Build a NamedTuple overriding selected variables with SingleRowOverrideVector wrappers
function build_row_override_data(base::NamedTuple, vars::Vector{Symbol}, row::Int)
    overrides = NamedTuple()
    override_vecs = Vector{SingleRowOverrideVector}(undef, length(vars))
    # Construct override vectors and merge into NamedTuple shadowing base
    pairs = Pair{Symbol,Any}[]
    for (i, s) in enumerate(vars)
        ov = SingleRowOverrideVector(getproperty(base, s), row, nothing)
        override_vecs[i] = ov
        push!(pairs, s => ov)
    end
    data_over = (; base..., pairs...)
    return data_over, override_vecs
end

# Callable closure for ForwardDiff that writes into a reusable buffer
struct DerivClosure{DE}
    de_ref::Base.RefValue{DE}
end

function (g::DerivClosure)(x::AbstractVector)
    de = g.de_ref[]
    Tx = eltype(x)
    # Select compiled + buffer (initialize dual on first use)
    if Tx === Float64
        compiled_T = de.compiled_base
        row_vec = de.rowvec_float
    else
        if de.compiled_dual === nothing
            UB = typeof(de.compiled_base)
            OpsT = UB.parameters[2]
            ST = UB.parameters[3]
            OT = UB.parameters[4]
            de.compiled_dual = UnifiedCompiled{Tx, OpsT, ST, OT}(de.compiled_base.ops)
            de.rowvec_dual = Vector{Tx}(undef, length(de))
        end
        compiled_T = de.compiled_dual
        row_vec = de.rowvec_dual
    end
    # Update overrides
    for i in eachindex(de.vars)
        ov = de.overrides[i]
        ov.row = de.row
        ov.replacement = x[i]
    end
    # Evaluate using prebuilt merged data
    compiled_T(row_vec, de.data_over, de.row)
    return row_vec
end

Base.length(g::DerivClosure) = length(g.de_ref[])

mutable struct DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged}
    compiled_base::UnifiedCompiled{T, Ops, S, O}
    base_data::NTBase
    vars::Vector{Symbol}
    xbuf::Vector{Float64}
    # Prebuilt row-local overrides and merged data (shared for Float64 and Dual)
    overrides::Vector{SingleRowOverrideVector}
    data_over::NTMerged
    # Row buffers
    rowvec_float::Vector{Float64}
    rowvec_dual::Any
    # Compiled dual instance
    compiled_dual::Any
    # AD closure and config
    g::Any
    cfg::Any
    row::Int
end

Base.length(de::DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged}) where {T, Ops, S, O, NTBase, NTMerged} = de.compiled_base |> length

"""
    build_derivative_evaluator(compiled, data; vars, chunk=:auto) -> DerivativeEvaluator

Build a ForwardDiff-based derivative evaluator for a fixed set of variables.

Arguments:
- `compiled::UnifiedCompiled`: Result of `compile_formula(model, data)`.
- `data::NamedTuple`: Column-table data (e.g., `Tables.columntable(df)`).
- `vars::Vector{Symbol}`: Variables to differentiate with respect to (typically continuous predictors).
- `chunk`: `ForwardDiff.Chunk{N}()` or `:auto` (uses `Chunk{length(vars)}`).

Returns:
- `DerivativeEvaluator`: Prebuilt evaluator object reusable across rows.

Notes:
- Compile once per model + variable set; reuse across calls.
- Zero allocations in steady state after warmup (typed closure + config; no per-call merges).
- Keep `vars` fixed for best specialization.
"""
function build_derivative_evaluator(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple;
    vars::Vector{Symbol},
    chunk=:auto,
) where {T, Ops, S, O}
    nvars = length(vars)
    xbuf = Vector{Float64}(undef, nvars)
    # Prebuild overrides + merged data (float path)
    data_over, overrides = build_row_override_data(data, vars, 1)
    rowvec_float = Vector{Float64}(undef, length(compiled))
    # Two-phase initialization to keep types concrete
    de_ref = Base.RefValue{DerivativeEvaluator{T, Ops, S, O, typeof(data), typeof(data_over)}}()
    g = DerivClosure(de_ref)
    # Choose chunk and build config using typed closure
    ch = chunk === :auto ? ForwardDiff.Chunk{nvars}() : chunk
    cfg = ForwardDiff.JacobianConfig(g, xbuf, ch)
    de = DerivativeEvaluator{T, Ops, S, O, typeof(data), typeof(data_over)}(
        compiled,
        data,
        vars,
        xbuf,
        overrides,
        data_over,
        rowvec_float,
        nothing,   # rowvec_dual
        nothing,   # compiled_dual
        g,
        cfg,
        1,
    )
    de_ref[] = de
    return de
end

"""
    derivative_modelrow!(J, deval, row) -> AbstractMatrix{Float64}

Fill `J` with the Jacobian of one model row with respect to `deval.vars`.

Arguments:
- `J::AbstractMatrix{Float64}`: Preallocated buffer of size `(n_terms, n_vars)`.
- `deval::DerivativeEvaluator`: Built by `build_derivative_evaluator`.
- `row::Int`: Row index (1-based).

Returns:
- The same `J` buffer, with `J[i, j] = ∂X[i]/∂vars[j]` for the given row.

Notes:
- Orientation is `(n_terms, n_vars)`; `n_terms == length(compiled)`.
- Zero allocations in steady state after warmup.
"""
function derivative_modelrow!(J::AbstractMatrix{Float64}, de::DerivativeEvaluator, row::Int)
    @assert size(J, 1) == length(de) "Jacobian row mismatch: expected $(length(de)) terms"
    @assert size(J, 2) == length(de.vars) "Jacobian column mismatch: expected $(length(de.vars)) variables"
    # Set row and seed x with base values
    de.row = row
    for (i, s) in enumerate(de.vars)
        de.xbuf[i] = getproperty(de.base_data, s)[row]
    end
    ForwardDiff.jacobian!(J, de.g, de.xbuf, de.cfg)
    return J
end

"""
    derivative_modelrow(deval, row) -> Matrix{Float64}

Allocating convenience wrapper that returns the Jacobian for one row.
"""
function derivative_modelrow(de::DerivativeEvaluator, row::Int)
    J = Matrix{Float64}(undef, length(de), length(de.vars))
    derivative_modelrow!(J, de, row)
    return J
end

############################# Finite Differences #############################

"""
    derivative_modelrow_fd!(J, compiled, data, row; vars, step=:auto)

Finite-difference Jacobian for a single row using central differences.

Arguments:
- `J::AbstractMatrix{Float64}`: Preallocated `(n_terms, n_vars)` buffer.
- `compiled::UnifiedCompiled`: Result of `compile_formula`.
- `data::NamedTuple`: Column-table data.
- `row::Int`: Row index.

Example:
```julia
g = Vector{Float64}(undef, length(deval.vars))
marginal_effects_eta!(g, deval, beta, 1)
```
- `vars::Vector{Symbol}`: Variables to differentiate with respect to.
- `step`: Numeric step size or `:auto` (`eps()^(1/3) * max(1, |x|)`).

Notes:
- Two evaluations per variable; useful as a robust fallback and for cross-checks.
"""
function derivative_modelrow_fd!(
    J::AbstractMatrix{Float64},
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    row::Int;
    vars::Vector{Symbol},
    step=:auto,
) where {T, Ops, S, O}
    @assert size(J, 1) == length(compiled)
    @assert size(J, 2) == length(vars)
    # Build row-local override once for all vars
    data_over, overrides = build_row_override_data(data, vars, row)
    # Buffers
    yplus = Vector{Float64}(undef, length(compiled))
    yminus = Vector{Float64}(undef, length(compiled))
    # Base values for each var at row
    xbase = similar(yplus, length(vars))
    for (j, s) in enumerate(vars)
        xbase[j] = getproperty(data, s)[row]
    end
    # Iterate variables
    for (j, s) in enumerate(vars)
        x = xbase[j]
        # Set all other overrides to base
        for (k, sk) in enumerate(vars)
            overrides[k].replacement = xbase[k]
        end
        # Step selection
        h = step === :auto ? (eps(Float64)^(1/3) * max(1.0, abs(float(x)))) : step
        # Plus
        overrides[j].replacement = x + h
        compiled(yplus, data_over, row)
        # Minus
        overrides[j].replacement = x - h
        compiled(yminus, data_over, row)
        # Central difference column
        @inbounds @fastmath for i in 1:length(compiled)
            J[i, j] = (yplus[i] - yminus[i]) / (2h)
        end
    end
    return J
end

function derivative_modelrow_fd(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    row::Int;
    vars::Vector{Symbol},
    step=:auto,
) where {T, Ops, S, O}
    J = Matrix{Float64}(undef, length(compiled), length(vars))
    derivative_modelrow_fd!(J, compiled, data, row; vars=vars, step=step)
    return J
end

############################## Discrete Contrasts ############################

"""
    contrast_modelrow!(Δ, compiled, data, row; var, from, to)

Compute a discrete contrast at one row for a single variable: `Δ = X(to) − X(from)`.

Arguments:
- `Δ::AbstractVector{Float64}`: Preallocated buffer of length `n_terms`.
- `compiled::UnifiedCompiled`: Result of `compile_formula`.
- `data::NamedTuple`: Column-table data.
- `row::Int`: Row index.

Example:
```julia
g = Vector{Float64}(undef, length(deval.vars))
marginal_effects_eta!(g, deval, beta, 1)
```
- `var::Symbol`: Variable to change (e.g., `:group3`).
- `from`, `to`: Values to contrast (level names or `CategoricalValue` for categorical; numbers for discrete).

Notes:
- Uses a row-local override; for categorical columns, values are normalized to the column’s levels.
"""
function contrast_modelrow!(
    Δ::AbstractVector{Float64},
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    row::Int;
    var::Symbol,
    from,
    to,
) where {T, Ops, S, O}
    @assert length(Δ) == length(compiled)
    # Build override wrapper for just this variable
    data_over, overrides = build_row_override_data(data, [var], row)
    # If categorical, ensure replacement is a CategoricalValue consistent with column levels
    base_col = getproperty(data, var)
    y_from = Vector{Float64}(undef, length(compiled))
    y_to = Vector{Float64}(undef, length(compiled))
    # From
    if (Base.find_package("CategoricalArrays") !== nothing) && (base_col isa CategoricalArrays.CategoricalArray)
        levels_list = levels(base_col)
        temp = CategoricalArrays.categorical([from], levels=levels_list, ordered=CategoricalArrays.isordered(base_col))
        overrides[1].replacement = temp[1]
    else
        overrides[1].replacement = from
    end
    compiled(y_from, data_over, row)
    # To
    if (Base.find_package("CategoricalArrays") !== nothing) && (base_col isa CategoricalArrays.CategoricalArray)
        levels_list = levels(base_col)
        temp = CategoricalArrays.categorical([to], levels=levels_list, ordered=CategoricalArrays.isordered(base_col))
        overrides[1].replacement = temp[1]
    else
        overrides[1].replacement = to
    end
    compiled(y_to, data_over, row)
    # Δ
    @inbounds @fastmath for i in 1:length(compiled)
        Δ[i] = y_to[i] - y_from[i]
    end
    return Δ
end

function contrast_modelrow(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    row::Int;
    var::Symbol,
    from,
    to,
) where {T, Ops, S, O}
    Δ = Vector{Float64}(undef, length(compiled))
    contrast_modelrow!(Δ, compiled, data, row; var=var, from=from, to=to)
    return Δ
end

# -- Marginal effects implementations (placed after types are defined) --

"""
    marginal_effects_eta!(g, deval, beta, row)

Fill `g` with marginal effects of η = Xβ w.r.t. `deval.vars` at `row`.
Implements: g = J' * β, where J = ∂X/∂vars.
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
)
    @assert length(g) == length(de.vars)
    @assert length(beta) == length(de)
    # Compute J once, then g = J' * beta
    J = Matrix{Float64}(undef, length(de), length(de.vars))
    derivative_modelrow!(J, de, row)
    mul!(g, transpose(J), beta)
    return g
end

function marginal_effects_eta(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta!(g, de, beta, row)
    return g
end

@inline _σ(x) = inv(1 + exp(-x))
const _INV_SQRT_2PI = 0.3989422804014327  # 1 / sqrt(2π)

@inline function _dmu_deta(link::GLM.IdentityLink, η::Real)
    return 1.0
end

@inline function _dmu_deta(link::GLM.LogLink, η::Real)
    return exp(η)  # μ = exp(η)
end

@inline function _dmu_deta(link::GLM.LogitLink, η::Real)
    μ = _σ(η)
    return μ * (1 - μ)  # σ'(η)
end

# Additional GLM links
@inline function _dmu_deta(link::GLM.ProbitLink, η::Real)
    # μ = Φ(η); dμ/dη = φ(η)
    return _INV_SQRT_2PI * exp(-0.5 * η^2)
end

@inline function _dmu_deta(link::GLM.CloglogLink, η::Real)
    # μ = 1 - exp(-exp(η)); dμ/dη = exp(η) * exp(-exp(η))
    return exp(η - exp(η))
end

@inline function _dmu_deta(link::GLM.CauchitLink, η::Real)
    # μ = (1/π) * atan(η) + 1/2; dμ/dη = 1 / (π * (1 + η^2))
    return inv(pi * (1 + η^2))
end

@inline function _dmu_deta(link::GLM.InverseLink, η::Real)
    # μ = 1/η; dμ/dη = -1 / η^2
    return -inv(η^2)
end

@inline function _dmu_deta(link::GLM.SqrtLink, η::Real)
    # μ = η^2; dμ/dη = 2η
    return 2 * η
end

# Some GLM variants include an inverse-square link (η = 1/μ^2)
if isdefined(GLM, :InverseSquareLink)
    @inline function _dmu_deta(link::GLM.InverseSquareLink, η::Real)
        # μ = η^(-1/2); dμ/dη = -(1/2) * η^(-3/2)
        return -0.5 * η^(-1.5)
    end
end

"""
    marginal_effects_mu!(g, deval, beta, row; link)

Compute marginal effects of μ = g⁻¹(η) at `row` via chain rule: dμ/dx = (dμ/dη) * (dη/dx).
Provide `link` (e.g., `IdentityLink()`, `LogLink()`, `LogitLink()`).
"""
function marginal_effects_mu!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    link=GLM.IdentityLink(),
)
    # Compute dη/dx
    gη = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta!(gη, de, beta, row)
    # Compute η at row
    xrow = Vector{Float64}(undef, length(de))
    de.compiled_base(xrow, de.base_data, row)
    η = dot(beta, xrow)
    scale = _dmu_deta(link, η)
    @inbounds @fastmath for j in eachindex(gη)
        g[j] = scale * gη[j]
    end
    return g
end

function marginal_effects_mu(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    link=GLM.IdentityLink(),
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_mu!(g, de, beta, row; link=link)
    return g
end


 
"""
    continuous_variables(compiled, data) -> Vector{Symbol}

Return a list of continuous variable symbols present in the compiled ops, excluding
categoricals detected via ContrastOps. Filters by `eltype(data[sym]) <: Real`.
"""
function continuous_variables(compiled::UnifiedCompiled, data::NamedTuple)
    cont = Set{Symbol}()
    cats = Set{Symbol}()
    for op in compiled.ops
        if op isa LoadOp
            Col = typeof(op).parameters[1]
            push!(cont, Col)
        elseif op isa ContrastOp
            Col = typeof(op).parameters[1]
            push!(cats, Col)
        end
    end
    # Remove any categorical columns
    for c in cats
        delete!(cont, c)
    end
    # Keep only columns that exist in data and are Real-typed
    vars = Symbol[]
    for s in cont
        if hasproperty(data, s)
            col = getproperty(data, s)
            if eltype(col) <: Real
                push!(vars, s)
            end
        end
    end
    sort!(vars)
    return vars
end
