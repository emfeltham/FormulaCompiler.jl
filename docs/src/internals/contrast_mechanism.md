# How Categorical Contrasts Work

This document explains the internal architecture and mechanism of categorical contrast computation in FormulaCompiler.jl, showing how the `ContrastEvaluator` achieves zero-allocation performance through the `CounterfactualVector` system.

## Table of Contents

1. [Overview](#overview)
2. [Simple Approach: Direct Data Modification](#simple-approach-direct-data-modification)
3. [CounterfactualVector Mechanism](#counterfactualvector-mechanism)
4. [ContrastEvaluator Initialization](#contrastevaluator-initialization)
5. [Contrast Computation Flow](#contrast-computation-flow)
6. [Gradient Computation](#gradient-computation)
7. [Performance Characteristics](#performance-characteristics)
8. [Advanced Topics](#advanced-topics)

---

## Overview

Categorical contrasts compute **counterfactual discrete differences**: comparing the same observation under different categorical levels.

```julia
# For a specific individual/observation at row i (notation, not code)
Δ = X(row=i, treatment="Drug_A") - X(row=i, treatment="Control")
```

This answers: "What would be the treatment effect for this specific individual, holding all other characteristics constant?"

**Important**: This is NOT comparing two different observations that happen to have different treatment levels (which would confound treatment effects with individual differences).

FormulaCompiler provides **two approaches** for computing these counterfactual contrasts:

### 1. Simple Approach: Direct Data Modification

For simple cases, you can modify data directly and compute differences:

```julia
# Evaluate with different categorical levels
data_control = merge(data, (treatment = fill("Control", n_rows),))
data_drug = merge(data, (treatment = fill("Drug_A", n_rows),))

X_control = modelrow(compiled, data_control, row)
X_drug = modelrow(compiled, data_drug, row)
Δ = X_drug - X_control
```

**Pros**: Simple, straightforward, no special APIs
**Cons**: Allocates new data structures, not suitable for batch processing

### 2. Zero-Allocation Approach: ContrastEvaluator + CounterfactualVectors

For performance-critical batch operations, use the zero-allocation system:

```julia
evaluator = contrastevaluator(compiled, data, [:treatment])
Δ = Vector{Float64}(undef, length(compiled))
contrast_modelrow!(Δ, evaluator, row, :treatment, "Control", "Drug_A")
```

**Pros**: Zero allocations, optimized for batch processing
**Cons**: More setup, requires understanding of evaluator pattern

The rest of this document focuses on **Approach 2** (the zero-allocation system).

---

## Simple Approach: Direct Data Modification

For exploratory analysis, one-off contrasts, or when performance isn't critical, you can compute contrasts by directly modifying data and evaluating the compiled formula twice.

!!! warning "Analytical Distinction: Counterfactual vs Cross-Sectional Contrasts"
    This approach computes **counterfactual contrasts**: comparing the **same observation** under different categorical levels. This is fundamentally different from comparing two different observations that happen to have different levels.

    **Counterfactual (what we're doing here)**:
    ```julia
    # Same person (row 5), different treatments
    X_row5_if_control = modelrow(compiled, data_control, 5)
    X_row5_if_drug = modelrow(compiled, data_drug, 5)
    Δ_counterfactual = X_row5_if_drug - X_row5_if_control
    ```
    This answers: "What would be the effect if this person received Drug instead of Control?"

    **Cross-sectional (NOT what we're doing)**:
    ```julia
    # Different people, different treatments
    X_person_in_control = modelrow(compiled, data, row_a)  # Person A (in Control)
    X_person_in_drug = modelrow(compiled, data, row_b)     # Person B (in Drug)
    Δ_cross_sectional = X_person_in_drug - X_person_in_control
    ```
    This confounds treatment effect with individual differences!

### Basic Pattern

```julia
using FormulaCompiler, GLM, DataFrames, Tables, LinearAlgebra

# Fit model and compile
df = DataFrame(
    y = randn(100),
    x = randn(100),
    treatment = rand(["Control", "Drug_A", "Drug_B"], 100)
)
model = lm(@formula(y ~ x * treatment), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Choose a row to analyze
row = 1

# Create modified data with different treatment levels
n_rows = length(data.y)
data_control = merge(data, (treatment = fill("Control", n_rows),))
data_drug = merge(data, (treatment = fill("Drug_A", n_rows),))

# Evaluate model rows with different levels
X_control = modelrow(compiled, data_control, row)
X_drug = modelrow(compiled, data_drug, row)

# Compute contrast
Δ = X_drug .- X_control

# Compute effect and standard error
β = coef(model)
effect = dot(β, Δ)

# Gradient for uncertainty quantification (linear scale)
∇β = Δ  # Parameter gradient: ∂(effect)/∂β = Δ
vcov_matrix = vcov(model)
se = sqrt(dot(∇β, vcov_matrix, ∇β))

println("Treatment effect: $effect ± $se")
```

### What This Does

1. **`merge(data, (treatment = fill("Control", n_rows),))`**
   - Creates a new NamedTuple with all columns from `data`
   - Replaces the `:treatment` column with a vector of all `"Control"` values
   - **Crucially**: Keeps all other covariates (x, age, etc.) at their original values for each row
   - **Memory**: Allocates a new treatment column (one pointer per row: about 0.9 KB for 100 rows)

2. **`modelrow(compiled, data_control, row)`**
   - Evaluates the compiled formula for row 1 using `data_control`
   - Returns the model matrix row X **for the same individual** (row 1) as if they had `treatment="Control"`
   - **All other characteristics** (x, age, etc.) remain as observed for row 1
   - **Memory**: Allocates a new output vector (112 bytes for the six-column model above)

3. **`X_drug .- X_control`**
   - Element-wise subtraction: the discrete effect vector
   - Shows the treatment effect **for this specific individual** with their specific covariate values
   - Holds everything else constant (ceteris paribus)

**Key insight**: Both `X_control` and `X_drug` are evaluated for the **same row** (same person/observation), just with different treatment assignments. This is a counterfactual comparison, not a comparison across different observations.

### Why Counterfactual Contrasts Matter

The counterfactual approach isolates the treatment effect by holding all other variables constant:

```julia
# Example: effect for one specific person
# (suppose row 15 has age=45, x=2.3, education="College" in a richer model)
row = 15

# Counterfactual contrast: Same person, different treatment
X_if_control = modelrow(compiled, data_control, 15)  # This person AS IF Control
X_if_drug = modelrow(compiled, data_drug, 15)        # This person AS IF Drug
Δ_counterfactual = X_if_drug - X_if_control

# Result: Pure treatment effect for THIS SPECIFIC person
# Holds constant: age=45, x=2.3, education="College"
effect = dot(β, Δ_counterfactual)
```

**Interpretation**: "If this 45-year-old college graduate with x=2.3 received Drug instead of Control, the predicted outcome would change by `effect`."

**Contrast with cross-sectional comparison** (wrong approach):
```julia
# Find someone in Control and someone in Drug
row_control = findfirst(==("Control"), data.treatment)  # Person A: age=45, x=2.3
row_drug = findfirst(==("Drug_A"), data.treatment)      # Person B: age=60, x=-1.5

X_person_control = modelrow(compiled, data, row_control)
X_person_drug = modelrow(compiled, data, row_drug)
Δ_wrong = X_person_drug - X_person_control

# PROBLEM: This confounds treatment with age difference (60 vs 45)
# and x difference (-1.5 vs 2.3)!
```

**Why this matters for marginal effects**:
- Discrete marginal effects measure **ceteris paribus** changes (holding everything else constant)
- Counterfactual contrasts implement this mathematically
- This is the standard definition in econometrics and causal inference

### Computing Scalar Effects and Gradients

To get a scalar treatment effect, multiply by coefficients. For uncertainty quantification, also compute the parameter gradient:

```julia
β = coef(model)

# Discrete effect on linear predictor (η scale)
effect_eta = dot(β, Δ)

# Parameter gradient for linear scale (needed for standard errors)
# For linear scale, the gradient IS the contrast vector
∇β_eta = Δ  # ∂(effect_eta)/∂β = Δ

# Standard error via delta method
vcov_matrix = vcov(model)
se_eta = sqrt(dot(∇β_eta, vcov_matrix, ∇β_eta))

# Confidence interval
ci_lower_eta = effect_eta - 1.96 * se_eta
ci_upper_eta = effect_eta + 1.96 * se_eta

println("Linear scale effect: $effect_eta ± $se_eta")
println("95% CI: [$ci_lower_eta, $ci_upper_eta]")
```

**Response scale (for GLM models with link functions):**

```julia
using GLM

# Compute linear predictors
η_control = dot(β, X_control)
η_drug = dot(β, X_drug)

# Apply link function
link = GLM.LogitLink()  # Example: logistic regression
μ_control = GLM.linkinv(link, η_control)
μ_drug = GLM.linkinv(link, η_drug)

# Discrete effect on response scale
effect_mu = μ_drug - μ_control

# Parameter gradient for response scale (chain rule)
# ∇β = g'(η_drug) × X_drug - g'(η_control) × X_control
g_prime_control = GLM.mueta(link, η_control)  # dμ/dη at control
g_prime_drug = GLM.mueta(link, η_drug)        # dμ/dη at drug
∇β_mu = g_prime_drug .* X_drug .- g_prime_control .* X_control

# Standard error via delta method
se_mu = sqrt(dot(∇β_mu, vcov_matrix, ∇β_mu))

# Confidence interval
ci_lower_mu = effect_mu - 1.96 * se_mu
ci_upper_mu = effect_mu + 1.96 * se_mu

println("Response scale effect: $effect_mu ± $se_mu")
println("95% CI: [$ci_lower_mu, $ci_upper_mu]")
```

**Why gradients matter:**
- Enable uncertainty quantification (standard errors, confidence intervals)
- Essential for hypothesis testing
- Required for proper statistical inference
- Delta method: SE = √(∇β' Σ ∇β) where Σ is the covariance matrix

### Multiple Rows or Levels

For multiple comparisons, use loops and include gradients for inference:

```julia
# Compare all levels to reference (with uncertainty)
reference_level = "Control"
other_levels = ["Drug_A", "Drug_B"]

for level in other_levels
    data_level = merge(data, (treatment = fill(level, n_rows),))
    X_level = modelrow(compiled, data_level, row)
    X_ref = modelrow(compiled, data_control, row)

    Δ = X_level .- X_ref
    effect = dot(β, Δ)

    # Gradient and standard error
    ∇β = Δ  # For linear scale
    se = sqrt(dot(∇β, vcov_matrix, ∇β))

    println("Effect of $level vs $reference_level: $effect ± $se")
end

# Analyze multiple rows (with confidence intervals)
rows_of_interest = [1, 10, 50]
for row in rows_of_interest
    X_control_row = modelrow(compiled, data_control, row)
    X_drug_row = modelrow(compiled, data_drug, row)
    Δ_row = X_drug_row .- X_control_row

    effect_row = dot(β, Δ_row)
    se_row = sqrt(dot(Δ_row, vcov_matrix, Δ_row))
    ci_lower = effect_row - 1.96 * se_row
    ci_upper = effect_row + 1.96 * se_row

    println("Row $row effect: $effect_row, 95% CI: [$ci_lower, $ci_upper]")
end
```

### When to Use This Approach

**Good for:**
- Exploratory analysis (quick checks, prototyping)
- One-off comparisons (single contrasts)
- Simple scripts where performance isn't critical
- Teaching/learning (simpler to understand)

**Not ideal for:**
- Batch processing (computing 1000+ contrasts)
- Performance-critical code (allocations add up)
- Production pipelines (want zero-allocation guarantees)

### Memory Cost Analysis

For a dataset with 1000 rows and the six-column model above:

**Per contrast computed:**
- New treatment column: about 8 KB (`fill("Control", 1000)` allocates 8,256 bytes: one pointer per row)
- Model row output: 112 bytes per `modelrow` call
- **Total per contrast**: about 8 KB per counterfactual column, plus the output vectors

**For 1000 contrasts** that each build new columns: on the order of 10 MB allocated

Compare to `ContrastEvaluator`: **0 bytes** per contrast after setup.

### Integration with Existing Data Workflows

This approach works naturally with DataFrames manipulation:

```julia
using DataFrames

# Create counterfactual datasets using DataFrames operations
df_control = transform(df, :treatment => (_ -> "Control") => :treatment)
df_drug = transform(df, :treatment => (_ -> "Drug_A") => :treatment)

# Convert to columntables and evaluate
data_control = Tables.columntable(df_control)
data_drug = Tables.columntable(df_drug)

X_control = modelrow(compiled, data_control, row)
X_drug = modelrow(compiled, data_drug, row)
Δ = X_drug .- X_control
```

### Complete Example: Logistic Regression with Uncertainty

For GLM models with link functions, compute both linear and response scale effects (new names, so the running example above is kept):

```julia
using FormulaCompiler, GLM, DataFrames, Tables, LinearAlgebra

# Fit logistic regression
df_logit = DataFrame(
    outcome = rand([0, 1], 100),
    age = rand(18:80, 100),
    treatment = rand(["Control", "Drug"], 100)
)
model_logit = glm(@formula(outcome ~ age * treatment), df_logit, Binomial(), LogitLink())
data_logit = Tables.columntable(df_logit)
compiled_logit = compile_formula(model_logit, data_logit)

# Counterfactual data
n_logit = length(data_logit.outcome)
data_logit_control = merge(data_logit, (treatment = fill("Control", n_logit),))
data_logit_drug = merge(data_logit, (treatment = fill("Drug", n_logit),))

# Analyze specific individual
row = 1
β_logit = coef(model_logit)
V_logit = vcov(model_logit)
link = LogitLink()

# Evaluate model rows
X0 = modelrow(compiled_logit, data_logit_control, row)
X1 = modelrow(compiled_logit, data_logit_drug, row)
ΔX = X1 .- X0

# Linear scale (log-odds) with uncertainty
effect_eta = dot(β_logit, ΔX)
se_eta = sqrt(dot(ΔX, V_logit, ΔX))
println("Log-odds effect: $effect_eta ± $se_eta")

# Response scale (probability) with uncertainty
η0 = dot(β_logit, X0)
η1 = dot(β_logit, X1)
μ0 = GLM.linkinv(link, η0)  # Probability if Control
μ1 = GLM.linkinv(link, η1)  # Probability if Drug
effect_mu = μ1 - μ0

# Gradient for response scale (chain rule)
∇β_mu = GLM.mueta(link, η1) .* X1 .- GLM.mueta(link, η0) .* X0
se_mu = sqrt(dot(∇β_mu, V_logit, ∇β_mu))

println("Probability effect: $effect_mu ± $se_mu")
println("  P(outcome=1|Control) = $μ0")
println("  P(outcome=1|Drug) = $μ1")
```

### Alternative: Using Dictionaries

For ad-hoc modifications without merge:

```julia
# Create modified data manually
data_control = (
    x = data.x,
    y = data.y,
    treatment = fill("Control", length(data.y))  # Override one column
)

data_drug = (
    x = data.x,
    y = data.y,
    treatment = fill("Drug_A", length(data.y))
)

X_control = modelrow(compiled, data_control, row)
X_drug = modelrow(compiled, data_drug, row)
Δ = X_drug .- X_control

# With gradient
β = coef(model)
effect = dot(β, Δ)
se = sqrt(dot(Δ, vcov(model), Δ))
```

### Summary: Simple vs Zero-Allocation

| Aspect | Simple Approach | ContrastEvaluator |
|--------|----------------|-------------------|
| **Ease of use** | Very simple | Requires setup |
| **Code clarity** | Clear intent | More abstraction |
| **Memory per contrast** | about 8 KB (1000 rows) | 0 bytes |
| **Best for** | A few contrasts | Many contrasts |
| **Setup cost** | None | One-time construction (about 6 μs and 5 KB in the benchmark below) |
| **Learning curve** | Minimal | Moderate |

**Recommendation**: Start with the simple approach for exploration. Switch to `ContrastEvaluator` when:
- You need to compute many contrasts
- Performance/memory is critical
- Building production pipelines

---


## CounterfactualVector Mechanism

The `ContrastEvaluator` achieves the same counterfactual comparison as the simple approach, but with zero allocations. It uses `CounterfactualVector`s to substitute a value for a single row without copying data.

**Key insight**: Like the simple approach, this compares the **same row** under different categorical levels. The only difference is performance optimization, not the analytical concept.

The examples in this section use a `CategoricalArray` column. These types are internal: apart from `CategoricalCounterfactualVector`, they are not exported, and user code normally reaches them only through `contrastevaluator`.

### Core Concept

A `CounterfactualVector` wraps an original data column and **intercepts access to a specific row**, returning a replacement value instead of the original:

```julia
using FormulaCompiler, CategoricalArrays

treatment = categorical(["Control", "Drug_A", "Drug_B", "Control"])

# Wrap the column; row 1 is replaced by the level of treatment[2] ("Drug_A")
cf_vec = CategoricalCounterfactualVector(treatment, 1, treatment[2])

# A categorical counterfactual returns reference codes into the column's pool
# (levels "Control", "Drug_A", "Drug_B" have codes 1, 2, 3):
cf_vec[1]  # → 0x00000002 ("Drug_A", counterfactual)
cf_vec[2]  # → 0x00000002 ("Drug_A", original)
cf_vec[3]  # → 0x00000003 ("Drug_B", original)
cf_vec[4]  # → 0x00000001 ("Control", original)
```

Returning the integer reference code rather than a `CategoricalValue` keeps reads allocation-free; the compiled formula resolves the code against the column's pool when it looks up the contrast-matrix row.

### Implementation

All `CounterfactualVector` types share this interface (excerpt from `src/compilation/counterfactual_vectors.jl`):

```julia
abstract type CounterfactualVector{T} <: AbstractVector{T} end

# Generic getindex implementation
@inline Base.@propagate_inbounds Base.getindex(v::CounterfactualVector, i::Int) =
    (i == v.row ? v.replacement : v.base[i])
```

**Key insight**: This is just a conditional branch: no array copying, no allocations.

### Typed Variants

Different data types have specialized implementations (excerpt):

```julia
# Numeric variables; the column is stored as B and read as T
mutable struct NumericCounterfactualVector{T<:Real, B<:Real} <: CounterfactualVector{T}
    base::Vector{B}           # Original data
    row::Int                  # Row index to override
    replacement::T            # Replacement value
end

# Categorical variables: the replacement is a reference code of type R (e.g. UInt32)
mutable struct CategoricalCounterfactualVector{T,R} <: CounterfactualVector{R}
    base::CategoricalArray{T,1,R}
    row::Int
    replacement::R
end

# Boolean variables
mutable struct BoolCounterfactualVector <: CounterfactualVector{Bool}
    base::Vector{Bool}
    row::Int
    replacement::Bool
end
```

Other columns (for example a `Vector{String}` holding categorical levels) use `TypedCounterfactualVector{T,V}`, and mixture columns use `CategoricalMixtureCounterfactualVector{T}`.

**Type stability**: Each variant has concrete types, enabling compiler optimizations.

### Mutable Updates

The vectors are **mutable**, so fields can be updated without allocations:

```julia
# Update which row to override
FormulaCompiler.update_counterfactual_row!(cf_vec, 3)

# Update the replacement value (a CategoricalValue is converted to its reference code)
FormulaCompiler.update_counterfactual_replacement!(cf_vec, treatment[1])  # "Control"

cf_vec[3]  # → 0x00000001 ("Control", counterfactual)
cf_vec[1]  # → 0x00000002 (original again)
```

This enables reusing the same `CounterfactualVector` for multiple contrasts.

---

## ContrastEvaluator Initialization

### Construction Process

When you create a `ContrastEvaluator`, it builds the complete counterfactual infrastructure (here for data and a model with `treatment`, `education` and `female` columns):

```julia
evaluator = contrastevaluator(compiled, data, [:treatment, :education, :female])
```

#### Step 1: Build Internal Counterfactual Structure

The `ContrastEvaluator` internally creates `CounterfactualVector` wrappers for the specified variables:

```julia
# Internal implementation creates:
# - A tuple of typed `CounterfactualVector` objects (one per variable)
# - A `NamedTuple` that merges original data with `CounterfactualVector` wrappers
```

**Example structure** (schematic):
```julia
# Original data
data = (
    x = [1.0, 2.0, 3.0, ...],
    outcome = [0.5, 1.2, 0.8, ...],
    treatment = categorical(["Control", "Drug_A", "Drug_B", ...]),
    education = categorical(["HS", "College", "HS", ...]),
    female = [0, 1, 1, 0, ...]
)

# Internal data_counterfactual structure
data_counterfactual = (
    x = data.x,                                    # Original (not in vars)
    outcome = data.outcome,                        # Original (not in vars)
    treatment = counterfactuals[1],                # CategoricalCounterfactualVector
    education = counterfactuals[2],                # CategoricalCounterfactualVector
    female = counterfactuals[3]                    # NumericCounterfactualVector{Int64}
)
```

**Key insight**: Variables not in `vars` use original columns; variables in `vars` are wrapped in `CounterfactualVector`s. The wrappers start with `row = 0`, so they shadow no row until a contrast is computed. If a categorical column's pool lacks a level the fitted model knows (e.g. a subset that dropped the reference level), the evaluator wraps an internal copy with the missing levels appended; the caller's data is not modified.

#### Step 2: Pre-compute Categorical Level Mappings

For each `CategoricalCounterfactualVector`, `contrastevaluator` records every level of the column's pool with its reference code, as a `CategoricalLevelMap` whose type carries the variable name and a tuple of `(level, code)` pairs (excerpt from `contrastevaluator` in `src/compilation/contrast_evaluator.jl`):

```julia
for (i, var) in enumerate(vars)
    cf_vec = counterfactuals[i]
    if cf_vec isa CategoricalCounterfactualVector
        levs = levels(cf_vec.base)
        R = typeof(cf_vec).parameters[2]           # reference type (UInt8, UInt32, ...)
        level_pairs = Tuple((level, R(ref_idx)) for (ref_idx, level) in enumerate(levs))
        push!(level_map_structs, CategoricalLevelMap{var, typeof(level_pairs)}(level_pairs))
    end
end
categorical_level_maps = Tuple(level_map_structs)
```

**Purpose**: Converting a string such as `"Control"` to a `CategoricalValue` allocates, and copying a `CategoricalValue` allocates too. The level maps store plain reference codes, which are bits types, and the lookup (`_find_categorical_level`) is a generated function that unrolls the search over the tuple, so resolving a level does not allocate.

**Example** (for a pool with levels `"Control"`, `"Drug_A"`, `"Drug_B"` and `UInt32` codes; the stored levels are the values `levels` returns, shown here as strings, and compare equal to them):
```julia
CategoricalLevelMap{:treatment, ...}((("Control", 0x00000001),
                                      ("Drug_A", 0x00000002),
                                      ("Drug_B", 0x00000003)))
```

#### Step 3: Allocate Buffers

```julia
# End of contrastevaluator (excerpt)
ContrastEvaluator(
    compiled,
    vars_tuple,                                  # Tuple(vars)
    var_map,                                     # NamedTuple: variable → counterfactual index
    data_counterfactual,
    counterfactuals,
    Vector{Float64}(undef, length(compiled)),    # y_from_buf
    Vector{Float64}(undef, length(compiled)),    # y_to_buf
    categorical_level_maps,
    Vector{Float64}(undef, length(compiled)),    # gradient_buffer
    Vector{Float64}(undef, length(compiled)),    # xrow_from_buf
    Vector{Float64}(undef, length(compiled)),    # xrow_to_buf
    1                                            # row
)
```

**One-time cost**: All memory allocation happens during construction. Runtime evaluation reuses these buffers.

---

## Contrast Computation Flow

### Overview

Computing a contrast involves:
1. Reset all counterfactuals, set the variable to the "from" level, evaluate, store the result
2. Reset, set the variable to the "to" level, evaluate, store the result
3. Compute the difference

Let's trace through an example step by step. The trace assumes `treatment` is a `CategoricalArray`. For a `Vector{String}` column, as in the simple-approach examples above, the wrapper is a `TypedCounterfactualVector{String}`, the replacement is the string itself, and no level map is involved.

### Example Call

```julia
contrast_modelrow!(Δ, evaluator, 1, :treatment, "Control", "Drug_A")
```

This computes `Δ = X(treatment="Drug_A") - X(treatment="Control")` for row 1. Each endpoint is computed by `level_modelrow!`:

```julia
# contrast_modelrow! (src/compilation/contrast_evaluator.jl)
level_modelrow!(evaluator.y_from_buf, evaluator, row, var, from)
level_modelrow!(evaluator.y_to_buf, evaluator, row, var, to)
Δ .= evaluator.y_to_buf .- evaluator.y_from_buf

# level_modelrow!
reset_all_counterfactuals!(evaluator.counterfactuals)       # every wrapper back to row = 0
update_counterfactual_for_var!(evaluator, var, row, level)
evaluator.compiled(out, evaluator.data_counterfactual, row)
reset_all_counterfactuals!(evaluator.counterfactuals)
```

Resetting before and after each endpoint makes every contrast independent of earlier calls.

### Step 1: Update Counterfactual to "from" Level

```julia
update_counterfactual_for_var!(evaluator, :treatment, 1, "Control")
```

`update_counterfactual_for_var!` is generated per evaluator type: the lookup of `:treatment` is unrolled into a chain of `if var === ...` branches with literal tuple indices, so the counterfactual vector has a concrete type. For that vector it does:

```julia
# 1. Point the CounterfactualVector for :treatment at the row
update_counterfactual_row!(cf_vec, 1)                     # cf_vec.row = 1

# 2. Resolve the level to a reference code with the pre-computed level map (no allocation)
code = _find_categorical_level(evaluator.categorical_level_maps, :treatment, "Control")

# 3. Store it (mutable field assignment)
update_counterfactual_replacement!(cf_vec, code)          # cf_vec.replacement = code
```

**Result**: The `CounterfactualVector` for `:treatment` now behaves as:
```julia
data_counterfactual.treatment[1]  # → code of "Control" (counterfactual)
data_counterfactual.treatment[2]  # → original code
data_counterfactual.treatment[3]  # → original code
# ... all other rows return original values
```

### Step 2: Evaluate Formula with "from" Level

```julia
evaluator.compiled(evaluator.y_from_buf, evaluator.data_counterfactual, row)
```

**What happens**:
1. The compiled formula evaluates row 1 using `data_counterfactual`
2. When it reads `data_counterfactual.treatment[1]`, it gets the code of `"Control"` (the counterfactual)
3. All other variables use their original values (no substitution)
4. Result: `y_from_buf` contains the model matrix row X₀ with `treatment="Control"`

**Visualization**:
```
Original data row 1:    treatment="Drug_A", education="HS", x=1.5
Counterfactual row 1:   treatment="Control", education="HS", x=1.5
                                    ↑
                            substituted value
```

### Step 3: Update Counterfactual to "to" Level

After a reset, the same call is made with the comparison level:

```julia
update_counterfactual_for_var!(evaluator, :treatment, 1, "Drug_A")
```

**Same process as Step 1**, but now:
```julia
cf_vec.row == 1                        # Same row
cf_vec.replacement == code("Drug_A")   # Different level
```

**Result**: The same `CounterfactualVector` now returns the code of `"Drug_A"` for row 1.

### Step 4: Evaluate Formula with "to" Level

```julia
evaluator.compiled(evaluator.y_to_buf, evaluator.data_counterfactual, row)
```

**Result**: `y_to_buf` contains the model matrix row X₁ with `treatment="Drug_A"`

### Step 5: Compute Discrete Effect

```julia
Δ .= evaluator.y_to_buf .- evaluator.y_from_buf
```

**Result**: `Δ = X₁ - X₀` (the discrete effect vector)

This is the **contrast vector** showing how each coefficient contributes to the treatment effect. After the call, `contrast_endpoints(evaluator)` returns `(y_from_buf, y_to_buf)`, i.e. X₀ and X₁.

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Reset; update CounterfactualVector: treatment[1] → "Control" │
│    - Mutable field assignment: cf_vec.replacement = code        │
│    - Uses pre-computed level map (0 allocations)                │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Evaluate formula with data_counterfactual                    │
│    - compiled(y_from_buf, data_counterfactual, row=1)           │
│    - Formula reads data_counterfactual.treatment[1]             │
│    - CounterfactualVector returns "Control" (not original)      │
│    - Result: y_from_buf = X₀ (model row with Control)           │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Reset; update CounterfactualVector: treatment[1] → "Drug_A"  │
│    - Same vector, new replacement field (0 allocations)         │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Evaluate formula with data_counterfactual (again)            │
│    - compiled(y_to_buf, data_counterfactual, row=1)             │
│    - CounterfactualVector now returns "Drug_A"                  │
│    - Result: y_to_buf = X₁ (model row with Drug_A)              │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Compute difference                                           │
│    - Δ .= y_to_buf .- y_from_buf                                │
│    - Result: Δ = X₁ - X₀ (discrete effect)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Gradient Computation

### Purpose

Parameter gradients enable **uncertainty quantification** via the delta method:

```julia
β = coef(model)
vcov_matrix = vcov(model)
∇β = Vector{Float64}(undef, length(compiled))

# Compute gradient ∇β where ∇β[i] = ∂(discrete_effect)/∂β[i]
contrast_gradient!(∇β, evaluator, row, :treatment, "Control", "Drug_A", β)

# Delta method standard error
se = sqrt(∇β' * vcov_matrix * ∇β)
```

### Linear Scale Gradients

For discrete effects on the linear predictor scale η = Xβ:

```
discrete_effect = η₁ - η₀ = (X₁'β) - (X₀'β) = (X₁ - X₀)'β
```

The gradient is simply:
```
∇β = ΔX = X₁ - X₀
```

**Implementation** (`_contrast_gradient_linear_scale!` in `src/compilation/contrast_evaluator.jl`, condensed):
```julia
# Reset, update to "from" level and evaluate
reset_all_counterfactuals!(evaluator.counterfactuals)
update_counterfactual_for_var!(evaluator, var, row, from)
evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)
# → xrow_from_buf = X₀

# Reset, update to "to" level and evaluate
reset_all_counterfactuals!(evaluator.counterfactuals)
update_counterfactual_for_var!(evaluator, var, row, to)
evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)
# → xrow_to_buf = X₁

# Gradient is the contrast vector
∇β .= evaluator.xrow_to_buf .- evaluator.xrow_from_buf  # ∇β = X₁ - X₀
reset_all_counterfactuals!(evaluator.counterfactuals)
```

### Response Scale Gradients

For discrete effects on the response scale μ = g⁻¹(η):

```
discrete_effect = μ₁ - μ₀ = g⁻¹(η₁) - g⁻¹(η₀)
```

By the chain rule:
```
∇β = g'(η₁) × X₁ - g'(η₀) × X₀
```

Where `g'(η) = dμ/dη` is the derivative of the inverse link.

**Implementation** (`_contrast_gradient_response_scale!`, condensed):
```julia
# Compute X₀ and η₀ = X₀'β
update_counterfactual_for_var!(evaluator, var, row, from)
evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)
η₀ = dot(β, evaluator.xrow_from_buf)

# Compute X₁ and η₁ = X₁'β
update_counterfactual_for_var!(evaluator, var, row, to)
evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)
η₁ = dot(β, evaluator.xrow_to_buf)

# Inverse-link derivatives
g_prime_η₀ = _dmu_deta(link, η₀)  # dμ/dη at η₀
g_prime_η₁ = _dmu_deta(link, η₁)  # dμ/dη at η₁

# Apply chain rule
∇β .= g_prime_η₁ .* evaluator.xrow_to_buf .- g_prime_η₀ .* evaluator.xrow_from_buf
```

The resets around each evaluation are as in the linear case.

### Supported Link Functions

`_dmu_deta` has analytic methods for:
- `IdentityLink`: g'(η) = 1
- `LogLink`: g'(η) = exp(η)
- `LogitLink`: g'(η) = exp(η) / (1 + exp(η))²
- `ProbitLink`: g'(η) = φ(η) (standard normal PDF)
- `CloglogLink`, `CauchitLink`, `InverseLink`, `SqrtLink`, `InverseSquareLink`, `PowerLink`, `NegativeBinomialLink`

Any other `GLM.Link` falls back to forward-mode automatic differentiation of `GLM.linkinv` (see `link_derivatives`).

---

## Performance Characteristics

### Zero Allocations Achieved Through

1. **Pre-allocated buffers**
   - All output buffers allocated once during construction
   - Reused across all contrast computations
   - Buffers: `y_from_buf`, `y_to_buf`, `xrow_from_buf`, `xrow_to_buf`, `gradient_buffer`

2. **Mutable CounterfactualVectors**
   - Fields updated in place (`row`, `replacement`)
   - No array copying or temporary allocations
   - Type-stable concrete types enable compiler optimizations

3. **Pre-computed categorical mappings**
   - Level → reference code pairs stored in tuples at construction
   - Lookups are unrolled at compile time and return bits values
   - Avoids level searches in the `CategoricalArray` at evaluation time

4. **Type specialization**
   - All CounterfactualVector types are concrete (not abstract)
   - Compiled formula is fully type-specialized
   - Variable lookup in the evaluator is unrolled per evaluator type, so there is no runtime dispatch on hot paths

### Memory Efficiency Comparison

**Traditional approach** (allocates O(n) memory):
```julia
using StatsModels

# Copy entire dataset and modify
data_control = deepcopy(data)
data_control.treatment .= "Control"  # Copied column overwritten
X₀ = modelmatrix(formula(model), DataFrame(data_control))

data_drug = deepcopy(data)
data_drug.treatment .= "Drug_A"
X₁ = modelmatrix(formula(model), DataFrame(data_drug))

Δ = X₁[row, :] - X₀[row, :]
```

**Memory used**: two copies of the data plus two full model matrices

**CounterfactualVector approach** (no per-contrast memory):
```julia
# Create evaluator once (one-time cost)
evaluator = contrastevaluator(compiled, data, [:treatment])
Δ = Vector{Float64}(undef, length(compiled))

# Compute contrast (zero allocations)
contrast_modelrow!(Δ, evaluator, row, :treatment, "Control", "Drug_A")
```

**Memory used**: 0 bytes per contrast after construction. Construction allocates the wrappers, level maps and buffers once; it does not copy the data columns (except a categorical column whose pool lacks some of the model's levels, as noted above).

### Timing Benchmarks

Measured with BenchmarkTools on an Apple M4 Pro (Julia 1.13.3), for `y ~ x * treatment` fitted with `lm` on 100 rows, `treatment` a three-level `CategoricalArray`:

```julia
# Construction (one-time cost)
@benchmark contrastevaluator($compiled, $data, [:treatment])
# median about 6 μs, 5.2 KB allocated

# Contrast computation (repeated operations)
@benchmark contrast_modelrow!($Δ, $evaluator, 1, :treatment, "Control", "Drug_A")
# median 40-50 ns, 0 bytes allocated

# Gradient computation (linear scale; with LogitLink() 60-70 ns)
@benchmark contrast_gradient!($∇β, $evaluator, 1, :treatment, "Control", "Drug_A", $β)
# median 40-50 ns, 0 bytes allocated
```

The ranges cover repeated runs. Absolute times depend on the machine and the formula.

### Batch Processing

Processing multiple contrasts reuses the same buffers:

```julia
evaluator = contrastevaluator(compiled, data, [:treatment])
Δ = Vector{Float64}(undef, length(compiled))

# Zero allocations for all iterations
for row in 1:100
    contrast_modelrow!(Δ, evaluator, row, :treatment, "Control", "Drug_A")
    # Process Δ...
end
```

**Performance**: Constant memory usage regardless of number of contrasts.

---

## Advanced Topics

### Multiple Variables

The same evaluator can process different variables (here for a model that also has a `region` factor with levels `"North"` and `"South"`):

```julia
evaluator = contrastevaluator(compiled, data, [:treatment, :region])
Δ_treatment = Vector{Float64}(undef, length(compiled))
Δ_region = Vector{Float64}(undef, length(compiled))

# Different variables, same evaluator, same buffers
contrast_modelrow!(Δ_treatment, evaluator, 1, :treatment, "Control", "Drug_A")
contrast_modelrow!(Δ_region, evaluator, 1, :region, "North", "South")
```

**Memory**: no allocation per contrast, whatever the number of variables or contrasts.

### Integration with Compiled Formula

The compiled formula is **unaware** that it is reading CounterfactualVectors (schematic):

```julia
# Inside the compiled formula evaluation
function (compiled::UnifiedCompiled)(output, data, row)
    # Read the column (may be a CounterfactualVector)
    treatment_val = data.treatment[row]  # Dispatches to getindex

    # The CounterfactualVector returns the replacement if row matches,
    # otherwise the original value; the formula sees no difference
end
```

**Transparency**: The `AbstractVector` interface makes CounterfactualVectors look like regular vectors to the compiled formula. The one special case is the categorical wrapper, whose elements are reference codes: the contrast operation resolves them against the wrapped column's pool.

### Type Stability Validation

All CounterfactualVector types have concrete element types:

```julia
# Numeric: Float64 in, Float64 out
cf_numeric = FormulaCompiler.counterfactualvector([1.0, 2.0, 3.0], 1)
typeof(cf_numeric)  # NumericCounterfactualVector{Float64, Float64}
eltype(cf_numeric)  # Float64

# Categorical: reference codes out
cf_cat = FormulaCompiler.counterfactualvector(categorical(["A", "B", "A"]), 1)
typeof(cf_cat)      # CategoricalCounterfactualVector{String, UInt32}
eltype(cf_cat)      # UInt32

# Boolean: Bool in, Bool out
cf_bool = FormulaCompiler.counterfactualvector([true, false], 1)
typeof(cf_bool)     # BoolCounterfactualVector
eltype(cf_bool)     # Bool
```

**Result**: No type instabilities, no runtime dispatch, full compiler optimization.

---

## Summary

The categorical contrast system achieves zero-allocation performance through:

1. **CounterfactualVectors**: Mutable wrappers that intercept single-row access
2. **Pre-computed mappings**: Categorical level lookups cached at construction
3. **Buffer reuse**: All output arrays allocated once, reused for all contrasts
4. **Type specialization**: Concrete types throughout enable compiler optimization

**The pattern**:
```
Construct evaluator → Update counterfactual → Evaluate → Update counterfactual → Evaluate → Difference
```

All with **zero allocations** after the one-time construction cost, so many contrasts can be computed without per-contrast memory.
