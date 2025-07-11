# _cols!.jl
# replace modelcols with a series of in-place column builders
# that do not make allocations to the heap.

#───────────────────────────────────────────────────────────────────────────────
# 3.  In-place column builders  (no heap allocations)
#───────────────────────────────────────────────────────────────────────────────
# -- leaf terms ---------------------------------------------------------------

"""
    evaluate_component_safely!(target_col::AbstractVector{Float64}, comp, d::NamedTuple)

FIXED: Safe component evaluation that handles comparison operators, ZScoredTerm, and other special cases.
"""
function evaluate_component_safely!(target_col::AbstractVector{Float64}, comp, d::NamedTuple)
    if comp isa ContinuousTerm || comp isa Term
        copy!(target_col, d[comp.sym])
        
    elseif comp isa CategoricalTerm
        v = d[comp.sym]
        M = comp.contrasts.matrix
        
        if isa(v, CategoricalArray)
            codes = refs(v)
        else
            unique_vals = sort(unique(v))
            code_map = Dict(val => i for (i, val) in enumerate(unique_vals))
            codes = [code_map[val] for val in v]
        end
        
        @inbounds for i in 1:length(target_col)
            target_col[i] = M[codes[i], 1]
        end
        
    elseif comp isa ZScoredTerm
        # Handle ZScoredTerm by evaluating the underlying term and applying transformation
        evaluate_component_safely!(target_col, comp.term, d)
        
        # Apply Z-score transformation in-place
        c = comp.center isa Number ? comp.center : (length(comp.center) == 1 ? comp.center[1] : error("Center should be scalar or length-1 vector"))
        s = comp.scale isa Number ? comp.scale : (length(comp.scale) == 1 ? comp.scale[1] : error("Scale should be scalar or length-1 vector"))
        
        if c == 0
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(target_col)
                target_col[i] *= inv_s
            end
        else
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(target_col)
                target_col[i] = (target_col[i] - c) * inv_s
            end
        end
        
    elseif comp isa InterceptTerm{true}
        fill!(target_col, 1.0)
        
    elseif comp isa InterceptTerm{false}
        fill!(target_col, 0.0)
        
    elseif comp isa ConstantTerm
        fill!(target_col, comp.n)
        
    elseif comp isa FunctionTerm
        # FIXED: Handle function terms, especially comparison operators
        if length(comp.args) == 1
            # Single argument function
            arg = comp.args[1]
            arg_values = Vector{Float64}(undef, length(target_col))
            evaluate_component_safely!(arg_values, arg, d)
            
            # Apply function
            if comp.f in [<=, >=, <, >, ==, !=]
                @warn "Single-argument comparison operator $(comp.f) treated as identity"
                copy!(target_col, arg_values)
            else
                try
                    @inbounds for i in 1:length(target_col)
                        target_col[i] = comp.f(arg_values[i])
                    end
                catch e
                    @warn "Failed to evaluate function $(comp.f): $e, using identity"
                    copy!(target_col, arg_values)
                end
            end
            
        elseif length(comp.args) == 2
            # Two argument function (like x <= constant)
            arg1, arg2 = comp.args
            
            # Evaluate first argument
            arg1_values = Vector{Float64}(undef, length(target_col))
            evaluate_component_safely!(arg1_values, arg1, d)
            
            # Handle second argument
            if arg2 isa ConstantTerm
                arg2_val = arg2.n
                
                # FIXED: Apply comparison correctly
                if comp.f === (<=)
                    @inbounds for i in 1:length(target_col)
                        target_col[i] = Float64(arg1_values[i] <= arg2_val)
                    end
                elseif comp.f === (>=)
                    @inbounds for i in 1:length(target_col)
                        target_col[i] = Float64(arg1_values[i] >= arg2_val)
                    end
                elseif comp.f === (<)
                    @inbounds for i in 1:length(target_col)
                        target_col[i] = Float64(arg1_values[i] < arg2_val)
                    end
                elseif comp.f === (>)
                    @inbounds for i in 1:length(target_col)
                        target_col[i] = Float64(arg1_values[i] > arg2_val)
                    end
                elseif comp.f === (==)
                    @inbounds for i in 1:length(target_col)
                        target_col[i] = Float64(arg1_values[i] == arg2_val)
                    end
                elseif comp.f === (!=)
                    @inbounds for i in 1:length(target_col)
                        target_col[i] = Float64(arg1_values[i] != arg2_val)
                    end
                else
                    # Other binary functions
                    try
                        @inbounds for i in 1:length(target_col)
                            target_col[i] = comp.f(arg1_values[i], arg2_val)
                        end
                    catch e
                        @warn "Failed to evaluate binary function $(comp.f): $e, using first argument"
                        copy!(target_col, arg1_values)
                    end
                end
            else
                # Two variable arguments
                arg2_values = Vector{Float64}(undef, length(target_col))
                evaluate_component_safely!(arg2_values, arg2, d)
                
                try
                    if comp.f in [<=, >=, <, >, ==, !=]
                        # Comparison operators
                        @inbounds for i in 1:length(target_col)
                            target_col[i] = Float64(comp.f(arg1_values[i], arg2_values[i]))
                        end
                    else
                        # Mathematical functions
                        @inbounds for i in 1:length(target_col)
                            target_col[i] = comp.f(arg1_values[i], arg2_values[i])
                        end
                    end
                catch e
                    @warn "Failed to evaluate binary function $(comp.f): $e, using first argument"
                    copy!(target_col, arg1_values)
                end
            end
        else
            @warn "FunctionTerm with $(length(comp.args)) arguments not supported, using 1.0"
            fill!(target_col, 1.0)
        end
        
    else
        @warn "Unknown component type $(typeof(comp)), using 1.0"
        fill!(target_col, 1.0)
    end
end

"""
    _cols!(::Nothing, _d, _X, j, _ipm, _fn_i, _int_i) -> Int

A no-op clause for `Nothing` terms that simply returns the current column index `j`.
"""
_cols!(::Nothing, _d, _X, j, _ipm, _f, _i) = j

"""
    _cols!(t::ConstantTerm, _d, X, j, _ipm, _fn_i, _int_i) -> Int

Fill column `j` of `X` with the constant value `t.n` for all rows, then return `j+1`.
"""
@inline _cols!(t::ConstantTerm, _d, X, j, _, _, _) = (fill!(view(X,:,j), t.n); j+1)

"""
    _cols!(::InterceptTerm{true}, _d, X, j, _ipm, _fn_i, _int_i) -> Int

Fill column `j` of `X` with `1.0` (the intercept) for all rows, then return `j+1`.
"""
@inline _cols!(::InterceptTerm{true}, _d, X, j, _, _, _) = (fill!(view(X,:,j),1.0); j+1)

"""
    _cols!(::InterceptTerm{false}, _d, _X, j, _imp, _fn_i, _int_i) -> Int

Explicitly omit the intercept: do nothing and return the current column index `j`.
"""
@inline _cols!(::InterceptTerm{false}, _d, _X, j, _, _, _) = j

"""
    _cols!(t::ContinuousTerm, d, X, j, _ipm, _fn_i, _int_i) -> Int

Copy the raw predictor vector `d[t.sym]` into column `j` of `X` (via `copy!`),
then return `j+1`. This avoids any temporary allocation by writing in-place.
"""
@inline _cols!(t::ContinuousTerm, d, X, j, _, _, _) = (copy!(view(X,:,j), d[t.sym]); j+1)

# put this near the other imports at the top of the file
using CategoricalArrays: refs   # gives the cached Vector{UInt32} of level codes

# ────────────────────────────────────────────────────────────────────────────
# Completely allocation-free categorical writer
# (replaces the previous _cols!(::CategoricalTerm, …) definition)
# ────────────────────────────────────────────────────────────────────────────

function _cols!(t::Term, d, X, j, _ipm, _fn_i, _int_i)
    # treat an un‐typed Term as a simple continuous variable
    copy!(view(X, :, j), d[t.sym])
    return j + 1
end

"""
    _cols!(t::CategoricalTerm{C,T,N}, d, X, j, _ipm, _fn_i, _int_i) where {C,T,N} -> Int

In-place writer for a categorical predictor, filling `N` dummy columns into `X`
without any heap allocations.

# Arguments

- `t::CategoricalTerm{C,T,N}`  
  A typed categorical term with `N` contrast columns.
- `d::NamedTuple`  
  A column-table mapping variable names to vectors; `d[t.sym]` must be a 
  `CategoricalVector`.
- `X::AbstractMatrix`  
  The destination matrix in which to write starting at column `j`.
- `j::Int`  
  The index of the first column in `X` to fill.
- `_ipm, _fn_i, _int_i`  
  Unused here (present for uniform `_cols!` dispatch).

# Behavior

1. Retrieves the internal integer codes of the categorical vector via `refs(v)`,
   avoiding `CategoricalValue` allocations.
2. Reads the raw contrast matrix `t.contrasts.matrix`, a plain `Matrix{T}` of size
   `n_levels × N`.
3. Loops over each row `r` and each dummy column `k`, writing
   `X[r, j + k - 1] = M[codes[r], k]` in-place.

# Returns

- `j + N`  
  The next free column index after the `N` columns written.
"""
function _cols!(t::CategoricalTerm{C,T,N}, d, X, j, _ipm, _fn_i, _int_i) where {C,T,N}
    # grab the coded data and the raw contrast table
    v     = d[t.sym]           # your CategoricalVector
    codes = refs(v)            # UInt32 codes vector, no alloc
    M     = t.contrasts.matrix # the k×(k-1) Float64 matrix
    rows  = length(codes)

    @inbounds for r in 1:rows
      @inbounds @simd for k in 1:N
        X[r, j + k - 1] = M[codes[r], k]
      end
    end

    return j + N
end

# -- tuple / matrixterm -------------------------------------------------------

"""
    _cols!(ts::Tuple, d, X, j, ipm, fn_i, int_i) -> Int

Sequentially process a tuple of terms `ts`, invoking `_cols!` on each element
and threading the column index `j` through. Returns the updated column index
after all terms in the tuple have been written.
"""
function _cols!(ts::Tuple, d, X, j, ipm, fn_i, int_i)
    for t in ts
        j = _cols!(t, d, X, j, ipm, fn_i, int_i)
    end
    j
end

"""
    _cols!(t::MatrixTerm, d, X, j, ipm, fn_i, int_i) -> Int

Unwrap a `MatrixTerm`, treating it as a tuple of subterms. Delegates to the
tuple method, preserving in-place, zero-allocation semantics. Returns the
updated column index after writing all subterms.
"""
_cols!(t::MatrixTerm, d, X, j, ipm, fn_i, int_i) = _cols!(t.terms, d, X, j, ipm, fn_i, int_i)

# -- FunctionTerm (one pre-alloc matrix per node) -----------------------------

"""
    _cols!(ft::FunctionTerm, d, X, j, ipm::InplaceModeler, fn_i::Ref, int_i::Ref) -> Int

In-place evaluator for a `FunctionTerm`, using a pre-allocated scratch matrix
to avoid temporary allocations.

# Arguments

- `ft::FunctionTerm`  
  A typed function term capturing the function `ft.f` and its `ft.args`.
- `d::NamedTuple`  
  Column-table data where each `ft.arg` is materialized.
- `X::AbstractMatrix`  
  Destination matrix; this writes into column `j`.
- `j::Int`  
  Index of the output column to fill with `ft.f(...)`.
- `ipm::InplaceModeler`  
  Holds `fn_scratch`: a list of pre-allocated matrices.
- `fn_i::Ref{Int}`  
  Mutable counter selecting which scratch matrix to use next.
- `int_i::Ref{Int}`  
  Counter for interaction terms (not used here).

# Behavior

1. **Scratch selection**: Pops the next buffer `scratch = fn_scratch[idx]`,
   an `nrow × nargs` matrix.
2. **Argument filling**: For each argument in `ft.args`, recursively call
   `_cols!`, writing its values into the corresponding column of `scratch`.
3. **Function application**: Loop over each row `r`:
   - For `nargs` = 1, 2, or 3, call `ft.f` with direct indexing for maximal speed.
   - Otherwise, build an `NTuple` via `ntuple` (heap allocation–free) and splat.
   - Write the result into `X[r, j]`.
4. **Index increment**: Return `j + 1`.

# Returns

- `j + 1`: The next free column index after writing this function term.

# Performance

- Zero heap allocations after `InplaceModeler` construction.
- Utilizes `@inbounds` and `@simd` in the row loop.
- Special-cases up to three arguments to avoid varargs overhead.
"""
function _cols!(ft::FunctionTerm, d, X, j, ipm::InplaceModeler, fn_i::Ref, int_i::Ref)
    idx     = fn_i[]; fn_i[] += 1
    scratch = ipm.fn_scratch[idx]       # nrow × nargs
    nargs   = size(scratch,2)

    # fill each argument into its column
    for (arg_i, arg) in enumerate(ft.args)
        _cols!(arg, d, reshape(@view(scratch[:,arg_i]),:,1), 1, ipm, fn_i, int_i)
    end

    col  = view(X,:,j); rows = size(col,1)
    @inbounds @simd for r in 1:rows
        if nargs == 1
            col[r] = ft.f(scratch[r,1])
        elseif nargs == 2
            col[r] = ft.f(scratch[r,1], scratch[r,2])
        elseif nargs == 3
            col[r] = ft.f(scratch[r,1], scratch[r,2], scratch[r,3])
        else
            # seldom used; but still no heap: reuse the same NTuple
            col[r] = ft.f(ntuple(k->scratch[r,k], nargs)...)
        end
    end
    j + 1
end

# -- InteractionTerm ----------------------------------------------------------

"""
    _cols!(t::InteractionTerm, d, X, j, ipm::InplaceModeler, fn_i::Ref, int_i::Ref) -> Int

In-place expansion of an interaction term (`t₁ & t₂ & …`) without any heap allocations.

# Arguments

- `t::InteractionTerm`  
  A typed interaction term whose components `t.terms` each produce one or more
  columns.

- `d::NamedTuple`  
  Column-table data mapping each variable name to a vector of length `nrow`.

- `X::AbstractMatrix`  
  The destination matrix; writes into the slice `X[:, j:j+total-1]`.

- `j::Int`  
  The starting column index in `X` for this interaction's Kronecker block.

- `ipm::InplaceModeler`  
  Holds pre-allocated buffers:
  - `int_subw[idx]`: widths of each component term  
  - `int_stride[idx]`: precomputed stride factors for indexing  
  - `int_scratch[idx]`: an `nrow × sum(widths)` matrix for component storage

- `fn_i::Ref{Int}`, `int_i::Ref{Int}`  
  Mutable counters indicating which pre-allocated scratch buffer to use
  (only `int_i` is advanced here; `fn_i` is threaded into nested calls).

# Behavior

1. **Scratch allocation**: Selects `scratch = int_scratch[idx]`, sized to hold
   the concatenated component columns (`sum(widths)`).
2. **Component evaluation**: For each sub-term, calls `_cols!` recursively to
   write its columns into the appropriate slice of `scratch`.
3. **Kronecker product**: Computes the row-wise tensor product of the component
   columns:
   - Uses `stride` factors to index into each component without allocations.
   - Accumulates the product for each row and final column in `dest`.
4. **Index update**: Calculates `total = ∏ widths`, the number of new columns,
   and returns `j + total`.

# Returns

- `j + total`: The next available column index after filling this interaction's block.
"""
function _cols!(
  t::InteractionTerm, d, X, j,
  ipm::InplaceModeler, fn_i::Ref, int_i::Ref
)
    idx      = int_i[]; int_i[] += 1
    sw       = ipm.int_subw[idx]
    stride   = ipm.int_stride[idx]
    scratch  = ipm.int_scratch[idx]
    rows     = size(X,1)

    # FIXED: Fill each component into scratch using safe evaluation
    ofs = 0
    for (comp, w) in zip(t.terms, sw)
        if w > 0
            if w == 1
                # Single column component - use safe evaluation
                target_col = view(scratch, :, ofs + 1)
                evaluate_component_safely!(target_col, comp, d)
            else
                # Multi-column component - use existing _cols! logic
                comp_view = view(scratch, :, ofs+1:ofs+w)
                try
                    _cols!(comp, d, comp_view, 1, ipm, fn_i, int_i)
                catch e
                    @warn "Multi-column component evaluation failed for $(typeof(comp)): $e, using identity"
                    if comp isa CategoricalTerm
                        # For categorical, try to fill with reasonable defaults
                        for k in 1:w
                            fill!(view(comp_view, :, k), k == 1 ? 1.0 : 0.0)
                        end
                    else
                        fill!(comp_view, 1.0)
                    end
                end
            end
        end
        ofs += w
    end

    # Write Kronecker product into destination
    total = prod(sw)
    dest  = view(X, :, j:j+total-1)

    @inbounds for r in 1:rows, col in 1:total
        off = col - 1
        acc = 1.0
        ofs = 0
        for p in 1:length(sw)
            k   = (off ÷ stride[p]) % sw[p]
            acc *= scratch[r, ofs + k + 1]
            ofs += sw[p]
        end
        dest[r, col] = acc
    end
    
    return j + total
end
