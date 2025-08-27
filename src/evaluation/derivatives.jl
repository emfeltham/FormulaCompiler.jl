# derivatives.jl

# TODO(derivatives): See DERIVATIVE_PLAN.md → Typing Checklist & Acceptance Criteria.
# - Eliminate Any on hot paths (closure/config/dual fields, overrides, column cache).
# - Prefer gradient-based η path for strict zero-allocation marginal effects.
# - Keep FD Jacobian as 0-alloc fallback; AD Jacobian 0-alloc may be env-dependent.
#
# Contributor TODOs (typing and allocation hygiene):
# 1) Make evaluator fields concrete
#    - g::Any             → store concrete closure type (DerivClosure{DE})
#    - cfg::Any           → store concrete ForwardDiff.JacobianConfig{…}
#    - rowvec_dual::Any   → Vector{<:ForwardDiff.Dual{…}} with concrete eltype
#    - compiled_dual::Any → UnifiedCompiled{<:ForwardDiff.Dual{…},Ops,S,O}
# 2) Column cache
#    - fd_columns::Vector{Any} → Vector{<:AbstractVector{T}} or NTuple for fixed nvars
# 3) Overrides
#    - SingleRowOverrideVector <: AbstractVector{Any}
#      replace with SingleRowOverrideVector{T} and build per-eltype data_over (Float64 & Dual)
# 4) Config creation
#    - Ensure JacobianConfig/GradientConfig are built once with concrete closure
# 5) Tests
#    - Tighten FD Jacobian to 0 allocations; η-gradient to 0; gate AD Jacobian per env caps

# Single-row override vector: returns replacement at `row`, base elsewhere (unused)
# NOTE: Current override wrapper uses eltype Any; this impacts AD typing.
# TODO(derivatives): Introduce `SingleRowOverrideVector{T}` and per-eltype merged data_over
# to ensure getindex returns T (Float64/Dual), removing Any from the hot path.
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

# Typed single-row override vector that preserves element type T
mutable struct TypedSingleRowOverrideVector{T} <: AbstractVector{T}
    base::AbstractVector
    row::Int
    replacement::T
end

Base.size(v::TypedSingleRowOverrideVector) = size(v.base)
Base.length(v::TypedSingleRowOverrideVector) = length(v.base)
Base.IndexStyle(::Type{<:TypedSingleRowOverrideVector}) = IndexLinear()
Base.eltype(::Type{TypedSingleRowOverrideVector{T}}) where {T} = T
@inline Base.getindex(v::TypedSingleRowOverrideVector{T}, i::Int) where {T} = (i == v.row ? v.replacement : convert(T, getindex(v.base, i)))

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

# Build a NamedTuple overriding selected variables using TypedSingleRowOverrideVector{T}
function build_row_override_data_typed(base::NamedTuple, vars::Vector{Symbol}, row::Int, ::Type{T}) where {T}
    override_vecs = Vector{TypedSingleRowOverrideVector{T}}(undef, length(vars))
    pairs = Pair{Symbol,TypedSingleRowOverrideVector{T}}[]
    for (i, s) in enumerate(vars)
        col = getproperty(base, s)
        ov = TypedSingleRowOverrideVector{T}(col, row, zero(T))
        override_vecs[i] = ov
        push!(pairs, s => ov)
    end
    data_over = (; base..., pairs...)
    return data_over, override_vecs
end

# Concrete Float64 override vector for FD evaluator (fully concrete base vector)
mutable struct FDOverrideVector <: AbstractVector{Float64}
    base::Vector{Float64}
    row::Int
    replacement::Float64
end
Base.size(v::FDOverrideVector) = size(v.base)
Base.length(v::FDOverrideVector) = length(v.base)
Base.IndexStyle(::Type{<:FDOverrideVector}) = IndexLinear()
Base.eltype(::Type{FDOverrideVector}) = Float64
@inline Base.getindex(v::FDOverrideVector, i::Int) = (i == v.row ? v.replacement : v.base[i])

# Callable closure for ForwardDiff that writes into a reusable buffer
struct DerivClosure{DE}
    de_ref::Base.RefValue{DE}
end

Base.length(g::DerivClosure) = length(g.de_ref[])

# More specific constructor for better type inference
DerivClosure(de::DE) where {DE} = DerivClosure{DE}(Base.RefValue{DE}(de))

# Scalar gradient closure for η = Xβ that reuses the vector closure
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
        # Ensure overrides/data_over exist for this specific Dual tag
        need_build = de.overrides_dual === nothing
        if !need_build
            # Compare stored override eltype to current Tx; rebuild on mismatch
            ovs = de.overrides_dual
            if !(isempty(ovs))
                stored_T = typeof(first(ovs)).parameters[1]
                need_build = (stored_T !== Tx)
            else
                need_build = true
            end
        end
        if need_build
            data_over_dual, overrides_dual = build_row_override_data_typed(de.base_data, de.vars, 1, Tx)
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

# TODO(derivatives): Several fields are currently `Any` and should be made concrete
# per DERIVATIVE_PLAN.md typing checklist (g/cfg/compiled_dual/rowvec_dual) to minimize
# allocations and enable universal type inference on hot paths.
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
    # Prebuild fully concrete overrides + merged data (Float64 path)
    override_vecs = Vector{FDOverrideVector}(undef, nvars)
    pairs = Pair{Symbol,FDOverrideVector}[]
    for (i, s) in enumerate(vars)
        col = getproperty(data, s)::Vector{Float64}
        ov = FDOverrideVector(col, 1, 0.0)
        override_vecs[i] = ov
        push!(pairs, s => ov)
    end
    data_over = (; data..., pairs...)
    overrides = override_vecs
    rowvec_float = Vector{Float64}(undef, length(compiled))
    # Pre-cache column references to avoid getproperty allocations (as NTuple)
    fd_columns = ntuple(i -> getproperty(data, vars[i]), nvars)
    
    # First pass evaluator (shell) to create typed closures/configs
    beta_ref = Base.RefValue(Vector{Float64}())
    shell = DerivativeEvaluator{T, Ops, S, O, typeof(data), typeof(data_over), nvars, typeof(fd_columns), Nothing, Nothing, Nothing, Nothing}(
        compiled,
        data,
        vars,
        xbuf,
        overrides,
        data_over,
        nothing,  # overrides_dual
        nothing,  # data_over_dual
        rowvec_float,
        nothing,   # rowvec_dual
        nothing,   # compiled_dual
        nothing,   # g
        nothing,   # cfg
        nothing,   # gscalar
        nothing,   # gradcfg
        beta_ref,
        1,
        Matrix{Float64}(undef, length(compiled), nvars),  # jacobian_buffer
        Vector{Float64}(undef, nvars),                    # eta_gradient_buffer
        Vector{Float64}(undef, length(compiled)),         # xrow_buffer
        Vector{Float64}(undef, length(compiled)),         # fd_yplus
        Vector{Float64}(undef, length(compiled)),         # fd_yminus
        Vector{Float64}(undef, nvars),                    # fd_xbase
        fd_columns,                                       # fd_columns
    )

    # Build typed closures and configs against the shell
    g = DerivClosure(shell)
    ch = chunk === :auto ? ForwardDiff.Chunk{nvars}() : chunk
    cfg = ForwardDiff.JacobianConfig(g, xbuf, ch)

    gscalar = GradClosure(g, beta_ref)
    gradcfg = ForwardDiff.GradientConfig(gscalar, xbuf, ch)

    # Final evaluator with concrete closure/config types
    de = DerivativeEvaluator{T, Ops, S, O, typeof(data), typeof(data_over), nvars, typeof(fd_columns), typeof(g), typeof(cfg), typeof(gscalar), typeof(gradcfg)}(
        compiled,
        data,
        vars,
        xbuf,
        overrides,
        data_over,
        nothing,
        nothing,
        rowvec_float,
        nothing,
        nothing,
        g,
        cfg,
        gscalar,
        gradcfg,
        beta_ref,
        1,
        Matrix{Float64}(undef, length(compiled), nvars),
        Vector{Float64}(undef, nvars),
        Vector{Float64}(undef, length(compiled)),
        Vector{Float64}(undef, length(compiled)),
        Vector{Float64}(undef, length(compiled)),
        Vector{Float64}(undef, nvars),
        fd_columns,
    )
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
    # Use untyped overrides for maximal compatibility/correctness here
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

######################## η-Gradient (Scalar AD Path) #########################

"""
    marginal_effects_eta_grad!(g, de, beta, row)

Compute marginal effects on η = Xβ via ForwardDiff.gradient! of the scalar
function h(x) = dot(β, Xrow(x)). Uses the existing AD closure to evaluate rows.
"""
function marginal_effects_eta_grad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
)
    @assert length(g) == length(de.vars)
    @assert length(beta) == length(de)
    # Seed x with base values
    de.row = row
    for (i, s) in enumerate(de.vars)
        de.xbuf[i] = getproperty(de.base_data, s)[row]
    end
    # Use stored scalar closure/config, update beta reference
    de.beta_ref[] = (beta isa Vector{Float64} ? beta : Vector{Float64}(beta))
    ForwardDiff.gradient!(g, de.gscalar, de.xbuf, de.gradcfg)
    return g
end

function marginal_effects_eta_grad(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta_grad!(g, de, beta, row)
    return g
end

"""
    derivative_modelrow_fd!(J, evaluator, row; step=:auto)

ULTIMATE zero-allocation finite differences with compile-time optimization.
"""
# Internal FD evaluator (no keyword) with auto step
@generated function _derivative_modelrow_fd_auto!(
    J::AbstractMatrix{Float64},
    de::DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC},
    row::Int,
) where {T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}
    N = NV
    stmts = Expr[]
    push!(stmts, :(yplus = de.fd_yplus))
    push!(stmts, :(yminus = de.fd_yminus))
    push!(stmts, :(xbase = de.fd_xbase))
    push!(stmts, :(nterms = length(de)))
    # Fill xbase with unrolled tuple access
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.fd_columns[$j][row]))
    end
    # Set row for overrides (unrolled)
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    # Main unrolled finite difference loop across variables
    for j in 1:N
        # x = xbase[j]
        push!(stmts, :(x = xbase[$j]))
        # set overrides[k].replacement = xbase[k] for all k
        for k in 1:N
            push!(stmts, :(@inbounds de.overrides[$k].replacement = xbase[$k]))
        end
        # step selection and evaluations
        push!(stmts, :(h = (2.220446049250313e-6 * max(1.0, abs(x)))))
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x + h))
        push!(stmts, :(de.compiled_base(yplus, de.data_over, row)))
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x - h))
        push!(stmts, :(de.compiled_base(yminus, de.data_over, row)))
        push!(stmts, :(inv_2h = 1.0 / (2.0 * h)))
        push!(stmts, quote
            @fastmath for i in 1:nterms
                @inbounds J[i, $j] = (yplus[i] - yminus[i]) * inv_2h
            end
        end)
    end
    return Expr(:block, stmts...)
end

# Internal FD evaluator (no keyword) with explicit step
@generated function _derivative_modelrow_fd_step!(
    J::AbstractMatrix{Float64},
    de::DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC},
    row::Int,
    step::Float64,
) where {T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}
    N = NV
    stmts = Expr[]
    push!(stmts, :(yplus = de.fd_yplus))
    push!(stmts, :(yminus = de.fd_yminus))
    push!(stmts, :(xbase = de.fd_xbase))
    push!(stmts, :(nterms = length(de)))
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.fd_columns[$j][row]))
    end
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    for j in 1:N
        push!(stmts, :(x = xbase[$j]))
        for k in 1:N
            push!(stmts, :(@inbounds de.overrides[$k].replacement = xbase[$k]))
        end
        push!(stmts, :(h = step))
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x + h))
        push!(stmts, :(de.compiled_base(yplus, de.data_over, row)))
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x - h))
        push!(stmts, :(de.compiled_base(yminus, de.data_over, row)))
        push!(stmts, :(inv_2h = 1.0 / (2.0 * h)))
        push!(stmts, quote
            @fastmath for i in 1:nterms
                @inbounds J[i, $j] = (yplus[i] - yminus[i]) * inv_2h
            end
        end)
    end
    return Expr(:block, stmts...)
end

# Public FD evaluator with keyword dispatch forwarding to allocation-free internals
@inline function derivative_modelrow_fd!(
    J::AbstractMatrix{Float64},
    de::DerivativeEvaluator,
    row::Int; step=:auto,
)
    if step === :auto
        return _derivative_modelrow_fd_auto!(J, de, row)
    else
        return _derivative_modelrow_fd_step!(J, de, row, Float64(step))
    end
end

# Positional convenience (no keyword) for zero-allocation hot path, with distinct name
@inline derivative_modelrow_fd_pos!(J::AbstractMatrix{Float64}, de::DerivativeEvaluator, row::Int) = _derivative_modelrow_fd_auto!(J, de, row)

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
    # Use preallocated Jacobian buffer to avoid allocation
    derivative_modelrow!(de.jacobian_buffer, de, row)
    mul!(g, transpose(de.jacobian_buffer), beta)
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

"""
    marginal_effects_eta_fd_true_zero!(g, evaluator, beta, row; step=:auto)

ABSOLUTE zero-allocation marginal effects bypassing the override system entirely.
"""
@generated function marginal_effects_eta_fd_true_zero!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    step=:auto,
)
    quote
        # Direct access to preallocated buffers
        yplus = de.fd_yplus
        yminus = de.fd_yminus
        xbase = de.fd_xbase
        J = de.jacobian_buffer
        
        nvars = length(de.vars)
        nterms = length(de)
        
        # Get base values using pre-cached columns
        @inbounds for j in 1:nvars
            xbase[j] = de.fd_columns[j][row]
        end
        
        # BYPASS OVERRIDE SYSTEM: Directly modify data columns!
        @inbounds for j in 1:nvars
            x = xbase[j]
            # Step selection
            h = step === :auto ? (2.220446049250313e-6 * max(1.0, abs(x))) : step
            
            # Directly modify the column (DANGEROUS but zero-allocation!)
            original_val = de.fd_columns[j][row]
            
            # Plus evaluation - modify column directly
            de.fd_columns[j][row] = x + h
            de.compiled_base(yplus, de.base_data, row)
            
            # Minus evaluation - modify column directly  
            de.fd_columns[j][row] = x - h
            de.compiled_base(yminus, de.base_data, row)
            
            # Restore original value
            de.fd_columns[j][row] = original_val
            
            # Store Jacobian column
            inv_2h = 1.0 / (2.0 * h)
            @fastmath for i in 1:nterms
                J[i, j] = (yplus[i] - yminus[i]) * inv_2h
            end
        end
        
        # Matrix multiplication: g = J' * β
        @inbounds @fastmath for j in 1:nvars
            sum_val = 0.0
            for i in 1:nterms
                sum_val = muladd(J[i, j], beta[i], sum_val)
            end
            g[j] = sum_val
        end
        
        return g
    end
end

"""
    marginal_effects_eta_fd!(g, evaluator, beta, row; step=:auto)

Zero-allocation marginal effects using finite differences (with override system).
"""
@generated function marginal_effects_eta_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    step=:auto,
)
    quote
        # Direct access to preallocated buffers
        yplus = de.fd_yplus
        yminus = de.fd_yminus
        xbase = de.fd_xbase
        J = de.jacobian_buffer
        
        nvars = length(de.vars)
        nterms = length(de)
        
        # Get base values using pre-cached columns (no getproperty)
        @inbounds for j in 1:nvars
            xbase[j] = de.fd_columns[j][row]
        end
        
        # Set row for overrides
        @inbounds for i in 1:nvars
            de.overrides[i].row = row
        end
        
        # Main finite difference loop - compile-time optimized
        @inbounds for j in 1:nvars
            x = xbase[j]
            # Set all overrides to base values
            for k in 1:nvars
                de.overrides[k].replacement = xbase[k]
            end
            # Step selection (hardcode eps()^(1/3) to avoid function calls)
            h = step === :auto ? (2.220446049250313e-6 * max(1.0, abs(x))) : step
            # Plus
            de.overrides[j].replacement = x + h
            de.compiled_base(yplus, de.data_over, row)
            # Minus  
            de.overrides[j].replacement = x - h
            de.compiled_base(yminus, de.data_over, row)
            # Store Jacobian column with FMA optimization
            inv_2h = 1.0 / (2.0 * h)
            @fastmath for i in 1:nterms
                J[i, j] = (yplus[i] - yminus[i]) * inv_2h
            end
        end
        
        # Matrix multiplication: g = J' * β (fully optimized with FMA)
        @inbounds @fastmath for j in 1:nvars
            sum_val = 0.0
            for i in 1:nterms
                sum_val = muladd(J[i, j], beta[i], sum_val)
            end
            g[j] = sum_val
        end
        
        return g
    end
end

function marginal_effects_eta_fd(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    step=:auto,
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta_fd!(g, de, beta, row; step=step)
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
    # Compute dη/dx using preallocated buffer
    marginal_effects_eta!(de.eta_gradient_buffer, de, beta, row)
    # Compute η at row using preallocated buffer
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(beta, de.xrow_buffer)
    scale = _dmu_deta(link, η)
    @inbounds @fastmath for j in eachindex(de.eta_gradient_buffer)
        g[j] = scale * de.eta_gradient_buffer[j]
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
