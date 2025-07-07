## EfficientModelMatrices.jl

Build **zero-allocation** model matrices for GLM.jl and MixedModels.jl in pure
Julia.

`EfficientModelMatrices` replaces the allocation-heavy
`StatsModels.modelmatrix` pipeline with a fully **in-place** implementation that

* pre-allocates one scratch buffer per `FunctionTerm` and `InteractionTerm`;
* copies continuous and categorical predictors column-wise without temporaries;
* evaluates nested functions and interactions in tight, bounds-checked loops;
* works for

  * standard OLS / GLM fits (`lm`, `glm`);
  * mixed-effects models (`LinearMixedModel`, `GeneralizedLinearMixedModel`)
    — only the **fixed** part of the formula is materialised.

The result: **zero heap allocations** per call, even for 10⁶-row data sets.



### Installation

```julia
pkg> add https://github.com/your-org/EfficientModelMatrices.jl
```

The only required dependencies are `StatsModels`, `Tables`, and (optionally)
`GLM` / `MixedModels` for fitting models.



### Quick start

```julia
using EfficientModelMatrices, StatsModels, DataFrames, GLM, Tables

# 1. Fit any StatsModels-compatible model
df = DataFrame(y = randn(10^6),
               x = randn(10^6),
               g = categorical(rand(["A","B","C"], 10^6)))
m  = lm(@formula(y ~ 1 + x + C(g) + x & C(g)), df)

# 2. One-time construction (≈ 80 tiny allocations)
ipm = InplaceModeler(m, nrow(df))

# 3. Pre-allocate the output and fill it in-place (0 allocations)
X = Matrix{Float64}(undef, nrow(df), width(formula(m).rhs))
modelmatrix!(ipm, Tables.columntable(df), X)
```

`X` is now identical to `StatsModels.modelmatrix(m)`, but produced without
touching the garbage collector.



### API

| Function                       | Purpose                                                                                                 |                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `InplaceModeler(model, nrows)` | Scan the model’s RHS, allocate per-term scratch buffers, return a reusable state object.                |                                                              |
| `modelmatrix!(ipm, data, X)`   | Fill the pre-allocated `X` with the model matrix for `data`; **zero allocations** after `ipm` is built. |                                                              |
| `fixed_effects_form(model)`    | Utility that strips \`( …                                                                               | … )`terms from a`MixedModel\` formula; identity for GLM/OLS. |



### Testing

```julia
pkg> test EfficientModelMatrices
```

The test suite builds large design matrices under three scenarios:

1. Linear model with categoricals & interactions
2. Nested function terms (`a * inv(b)`, `log(abs(a-b))`, …)
3. Mixed-effects model (fixed-effects block only)

Each compares the in-place matrix exactly (`≈`) to the reference from
`StatsModels.modelmatrix` and asserts **zero allocations** with `@allocated`.

### Caveats

* Only the **fixed** part of a mixed model is supported; random-effects
  structures are ignored.
* Function terms with **> 3** arguments fall back to a var-args call; still
  allocation-free but marginally slower.
* Requires that `data` be a column table (`Tables.columntable(df)`).

