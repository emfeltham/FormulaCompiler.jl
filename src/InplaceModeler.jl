# InplaceModeler.jl

#───────────────────────────────────────────────────────────────────────────────
# 1.  State object with every scratch buffer pre-allocated once
#───────────────────────────────────────────────────────────────────────────────

"""
    InplaceModeler{M}

A helper object that encapsulates a fitted `StatisticalModel` and
all pre‐allocated scratch buffers needed to build its model matrix
in-place without any heap allocations.

# Type Parameters

- `M`: a subtype of `StatisticalModel` (e.g., a regression model type).

# Fields

- `model::M`  
  The fitted model whose formula drives the column‐generation logic.

- `fn_terms::Vector{FunctionTerm}`  
  All `FunctionTerm` nodes in the model’s RHS term tree, in encounter order.

- `fn_scratch::Vector{Matrix{Float64}}`  
  One scratch matrix per `FunctionTerm`, each of size `nrow × nargs`,
  used to hold the broadcast arguments before applying the function in-place.

- `int_terms::Vector{InteractionTerm}`  
  All `InteractionTerm` nodes in the model’s RHS term tree, in encounter order.

- `int_subw::Vector{Vector{Int}}`  
  For each `InteractionTerm`, a vector of component widths (`width(term_i)`).

- `int_stride::Vector{Vector{Int}}`  
  For each interaction, the cumulative-product “stride” array used
  to compute Kronecker‐style indices without allocations.

- `int_prefix::Vector{Vector{Int}}`  
  For each interaction, the rolling‐sum “prefix” offsets of component
  blocks within the per-interaction scratch matrix.

- `int_scratch::Vector{Matrix{Float64}}`  
  One scratch matrix per `InteractionTerm`, each of size
  `nrow × sum(int_subw[i])`, used to hold component columns
  before forming the row‐wise tensor product.
"""
mutable struct InplaceModeler{M}
    model       :: M
    fn_terms    :: Vector{FunctionTerm}
    fn_scratch  :: Vector{Matrix{Float64}}          # nrow × nargs for *each* fn
    int_terms   :: Vector{InteractionTerm}
    int_subw    :: Vector{Vector{Int}}              # widths of components
    int_stride  :: Vector{Vector{Int}}              # cumulative products
    int_prefix  :: Vector{Vector{Int}}              # column offsets
    int_scratch :: Vector{Matrix{Float64}}          # nrow × total columns
end

# ── recursive collector ──
"""
    _collect!(t, ::Type{T}, buf::Vector{T}) where T

Recursively traverse the term‐tree rooted at `t`, collecting all nodes of type `T`
into the provided buffer `buf`.

# Arguments

- `t`: An `AbstractTerm` (or tuple/formula/matrix‐term) to traverse.
- `::Type{T}`: The term type to collect (e.g., `FunctionTerm` or `InteractionTerm`).
- `buf::Vector{T}`: A vector into which matching terms are `push!`ed.

# Returns

- `buf`: The same vector passed in, now containing all encountered terms of type `T`
  in *preorder* (parent before children), in the order they were discovered.

# Behavior

- If `t isa T`, `t` is appended to `buf`.
- For composite terms (`FunctionTerm`, `InteractionTerm`, `Tuple`, `MatrixTerm`,
  `FormulaTerm`), it descends into each sub‐term to continue the search.
- Does **not** allocate new buffers; reuses `buf` in-place.
"""
function _collect!(t, ::Type{T}, buf::Vector{T}) where T
    t isa T && push!(buf, t)
    if     t isa FunctionTerm      ; foreach(a -> _collect!(a,T,buf), t.args)
    elseif t isa InteractionTerm   ; foreach(a -> _collect!(a,T,buf), t.terms)
    elseif t isa Tuple             ; foreach(a -> _collect!(a,T,buf), t)
    elseif t isa MatrixTerm        ; _collect!(t.terms, T, buf)
    elseif t isa FormulaTerm
        _collect!(t.lhs, T, buf);  _collect!(t.rhs, T, buf)
    end
    buf
end

# ── constructor ──
"""
    InplaceModeler(model::StatisticalModel, nrows::Int) -> InplaceModeler{<:StatisticalModel}

Create an `InplaceModeler` tailored to a fitted `model` and a fixed number of rows `nrows`,
pre-allocating all necessary scratch buffers for zero-allocation model matrix construction.

# Arguments

- `model::StatisticalModel`  
  A fitted statistical model (e.g., a `TableRegressionModel`) whose formula’s RHS
  defines the terms to materialize.

- `nrows::Int`  
  The number of observations (rows) in any new dataset you will later pass to
  `modelmatrix!(ipm, data, X)`.  All internal scratch matrices are sized to this.

# Returns

- `InplaceModeler{M}`  
  An object containing:
  1. `fn_terms` and `fn_scratch` — a list of each `FunctionTerm` node in the model’s
     formula and a corresponding `nrows × nargs` buffer.
  2. `int_terms`, `int_subw`, `int_stride`, `int_prefix`, and `int_scratch` — for each
     `InteractionTerm`, the component widths, precomputed strides and offsets, and
     an `nrows × sum(widths)` buffer.

Once constructed, the `InplaceModeler` can be reused to fill different datasets into
a pre-allocated output `X` matrix without any further heap allocations.
"""
function InplaceModeler(model::StatsModels.StatisticalModel, nrows::Int)
    rhs = formula(model).rhs

    fn_terms   = FunctionTerm[]
    _collect!(rhs, FunctionTerm, fn_terms)
    fn_scratch = [Matrix{Float64}(undef, nrows, length(ft.args)) for ft in fn_terms]

    int_terms  = InteractionTerm[]
    _collect!(rhs, InteractionTerm, int_terms)
    int_subw   = [[width(p) for p in it.terms] for it in int_terms]
    int_prefix = [cumsum([0; sw[1:end-1]]) for sw in int_subw]
    int_stride = [begin
                      s = Vector{Int}(undef, length(sw))
                      s[1] = 1
                      for k in 2:length(sw) s[k] = s[k-1]*sw[k-1] end
                      s
                  end for sw in int_subw]
    int_scratch = [Matrix{Float64}(undef, nrows, sum(sw)) for sw in int_subw]

    InplaceModeler(model, fn_terms, fn_scratch,
                   int_terms, int_subw, int_stride, int_prefix, int_scratch)
end
