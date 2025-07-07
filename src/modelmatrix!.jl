# modelmatrix!.jl

#───────────────────────────────────────────────────────────────────────────────
# 2.  Public driver
#───────────────────────────────────────────────────────────────────────────────

"""
    modelmatrix!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix) -> AbstractMatrix

Fill the pre-allocated matrix `X` with the model matrix corresponding to
`ipm.model` evaluated on the new dataset `data`, reusing internal scratch
buffers to avoid heap allocations.

# Arguments

- `ipm::InplaceModeler`  
  An `InplaceModeler` instance constructed for a specific fitted model and
  a fixed number of rows.  It carries pre-allocated buffers for every
  `FunctionTerm` and `InteractionTerm` in the model’s formula.

- `data::NamedTuple`  
  A table in “column-table” form (e.g. a `Tables.columntable(df)`),
  mapping each predictor name to a vector of length `n`.  These vectors
  are read directly to build each column of the output.

- `X::AbstractMatrix`  
  A pre-allocated `n × p` matrix of the correct size:
  - `n` must equal `nrow(data)`  
  - `p` must equal `width(formula(ipm.model).rhs)`  
  The contents of `X` will be overwritten in place.

# Returns

- `X`  
  The same matrix passed in, now containing the model matrix.  Useful for
  chaining or inlining within performance-sensitive loops.

# Behavior & Performance

- **Zero allocations** after the one-time construction of `InplaceModeler`.  
- Recursively walks the model’s RHS term tree, filling each column (and
  combinations of columns) directly into `X`.  
- Maintains two small counters (`fn_i`, `int_i`) to reuse the correct
  scratch buffer for each `FunctionTerm` and `InteractionTerm`.  
- Performs no temporary allocations: continuous predictors are
  `copy!`’d, categorical predictors are read via integer codes,
  functions are applied in tight loops, and interactions are built with
  nested loops over pre-computed strides.

# Throws

- `ArgumentError` if the pre-allocated `X` has the wrong number of columns.

# Example

```julia
using EfficientModeler, StatsModels, DataFrames, Tables

df       = DataFrame(x = 1:5, z = repeat(["a","b"], 5), y = rand(5))
f        = @formula(y ~ 1 + x + C(z) + x & C(z))
schema   = schema(f, df)
f_typed  = apply_schema(f, schema)
m        = lm(f_typed, df)

ipm = InplaceModeler(m, nrow(df))           # one-time, small allocation
X   = similar(modelmatrix(m))               # pre-allocate output

@time modelmatrix!(ipm, Tables.columntable(df), X)  # 0 allocations, in-place
"""
function modelmatrix!(ipm::InplaceModeler, data::NamedTuple, X::AbstractMatrix)
    rhs  = formula(ipm.model).rhs
    @assert width(rhs) == size(X,2) "pre-allocated X has wrong #cols"
    fn_i  = Ref(1)            # walk fn_terms in encounter order
    int_i = Ref(1)            # walk int_terms in encounter order
    _cols!(rhs, data, X, 1, ipm, fn_i, int_i)
    return X
end
