# evaluator.jl - DerivativeEvaluator construction and setup

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
        col = getproperty(data, s)
        # Convert integer columns to Float64 for derivative computation
        float_col = if col isa Vector{Int64} || col isa Vector{Int32} || col isa Vector{Int}
            convert(Vector{Float64}, col)
        else
            col::Vector{Float64}
        end
        ov = FDOverrideVector(float_col, 1, 0.0)
        override_vecs[i] = ov
        push!(pairs, s => ov)
    end
    # Merge with converted columns - use Float64 versions for derivative computation
    data_over = merge(data, NamedTuple(pairs))
    overrides = override_vecs
    rowvec_float = Vector{Float64}(undef, length(compiled))
    # Pre-cache column references to avoid getproperty allocations (as NTuple)
    # Use the converted Float64 columns for FD computation
    fd_columns = ntuple(i -> getproperty(data_over, vars[i]), nvars)
    
    # Create a mutable ref that will eventually point to the final evaluator
    beta_ref = Base.RefValue(Vector{Float64}())
    de_ref = Base.RefValue{DerivativeEvaluator}()
    
    # Build typed closures and configs using the ref (which is still uninitialized)
    g = DerivClosure(de_ref)
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
    
    # Now initialize the ref to point to the final evaluator
    de_ref[] = de
    
    return de
end