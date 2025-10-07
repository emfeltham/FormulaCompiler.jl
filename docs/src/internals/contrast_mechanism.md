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
# For a specific individual/observation at row i
Δ = X(row=i, treatment="Drug") - X(row=i, treatment="Control")
```

This answers: "What would be the treatment effect for this specific individual, holding all other characteristics constant?"

**Important**: This is NOT comparing two different observations that happen to have different treatment levels (which would confound treatment effects with individual differences).

FormulaCompiler provides **two approaches** for computing these counterfactual contrasts:

### 1. Simple Approach: Direct Data Modification

For simple cases, you can modify data directly and compute differences:

```julia
# Evaluate with different categorical levels
data_control = merge(data, (treatment = fill("Control", n_rows),))
data_drug = merge(data, (treatment = fill("Drug", n_rows),))

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
contrast_modelrow!(Δ, evaluator, row, :treatment, "Control", "Drug")
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
using FormulaCompiler, GLM, DataFrames, Tables

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
   - **Memory**: Allocates new vector for treatment column (~100 bytes for 100 rows)

2. **`modelrow(compiled, data_control, row)`**
   - Evaluates the compiled formula for row 1 using `data_control`
   - Returns the model matrix row X **for the same individual** (row 1) as if they had `treatment="Control"`
   - **All other characteristics** (x, age, etc.) remain as observed for row 1
   - **Memory**: Allocates new output vector (~80 bytes for typical model)

3. **`X_drug .- X_control`**
   - Element-wise subtraction: the discrete effect vector
   - Shows the treatment effect **for this specific individual** with their specific covariate values
   - Holds everything else constant (ceteris paribus)

**Key insight**: Both `X_control` and `X_drug` are evaluated for the **same row** (same person/observation), just with different treatment assignments. This is a counterfactual comparison, not a comparison across different observations.

### Why Counterfactual Contrasts Matter

The counterfactual approach isolates the treatment effect by holding all other variables constant:

```julia
# Example: Effect for a 45-year-old with specific characteristics
row = 15  # Person with: age=45, x=2.3, education="College"

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
row_control = findfirst(data.treatment .== "Control")  # Person A: age=45, x=2.3
row_drug = findfirst(data.treatment .== "Drug")        # Person B: age=60, x=-1.5

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

For a dataset with 1000 rows and a typical model:

**Per contrast computed:**
- New treatment vector: ~8KB (1000 × Float64)
- Model row output: ~80 bytes (typical)
- Intermediate allocations: ~100 bytes
- **Total per contrast**: ~8.2KB

**For 1000 contrasts**: ~8.2MB allocated

Compare to `ContrastEvaluator`: **0 bytes** per contrast after setup.

### Integration with Existing Data Workflows

This approach works naturally with DataFrames manipulation:

```julia
using DataFrames, Chain

# Create counterfactual datasets using DataFrames operations
df_control = @chain df begin
    transform(:treatment => (_ -> "Control") => :treatment)
end

df_drug = @chain df begin
    transform(:treatment => (_ -> "Drug_A") => :treatment)
end

# Convert to columntables and evaluate
data_control = Tables.columntable(df_control)
data_drug = Tables.columntable(df_drug)

X_control = modelrow(compiled, data_control, row)
X_drug = modelrow(compiled, data_drug, row)
Δ = X_drug .- X_control
```

### Complete Example: Logistic Regression with Uncertainty

For GLM models with link functions, compute both linear and response scale effects:

```julia
using GLM, DataFrames, Tables

# Fit logistic regression
df = DataFrame(
    outcome = rand([0, 1], 100),
    age = rand(18:80, 100),
    treatment = rand(["Control", "Drug"], 100)
)
model = glm(@formula(outcome ~ age * treatment), df, Binomial(), LogitLink())
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Counterfactual data
n_rows = length(data.outcome)
data_control = merge(data, (treatment = fill("Control", n_rows),))
data_drug = merge(data, (treatment = fill("Drug", n_rows),))

# Analyze specific individual
row = 1
β = coef(model)
vcov_matrix = vcov(model)
link = LogitLink()

# Evaluate model rows
X_control = modelrow(compiled, data_control, row)
X_drug = modelrow(compiled, data_drug, row)
Δ = X_drug .- X_control

# Linear scale (log-odds) with uncertainty
effect_eta = dot(β, Δ)
∇β_eta = Δ
se_eta = sqrt(dot(∇β_eta, vcov_matrix, ∇β_eta))
println("Log-odds effect: $effect_eta ± $se_eta")

# Response scale (probability) with uncertainty
η_control = dot(β, X_control)
η_drug = dot(β, X_drug)
μ_control = GLM.linkinv(link, η_control)  # Probability if Control
μ_drug = GLM.linkinv(link, η_drug)        # Probability if Drug
effect_mu = μ_drug - μ_control

# Gradient for response scale (chain rule)
g_prime_control = GLM.mueta(link, η_control)
g_prime_drug = GLM.mueta(link, η_drug)
∇β_mu = g_prime_drug .* X_drug .- g_prime_control .* X_control
se_mu = sqrt(dot(∇β_mu, vcov_matrix, ∇β_mu))

println("Probability effect: $effect_mu ± $se_mu")
println("  P(outcome=1|Control) = $μ_control")
println("  P(outcome=1|Drug) = $μ_drug")
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
| **Memory per contrast** | ~8KB | 0 bytes |
| **Best for** | 1-10 contrasts | 100+ contrasts |
| **Setup cost** | None | ~50μs + ~50KB |
| **Learning curve** | Minimal | Moderate |

**Recommendation**: Start with the simple approach for exploration. Switch to `ContrastEvaluator` when:
- You need to compute 100+ contrasts
- Performance/memory is critical
- Building production pipelines

---

## CounterfactualVector Mechanism

The `ContrastEvaluator` achieves the same counterfactual comparison as the simple approach, but with zero allocations. It uses `CounterfactualVector`s to efficiently substitute values for a single row without copying data.

**Key insight**: Like the simple approach, this compares the **same row** under different categorical levels. The only difference is performance optimization, not the analytical concept.

### Core Concept

A `CounterfactualVector` wraps an original data column and **intercepts access to a specific row**, returning a replacement value instead of the original:

```julia
# Original data column
data.treatment = ["Control", "Drug_A", "Drug_B", "Control", ...]

# Create CounterfactualVector
cf_vec = CategoricalCounterfactualVector(data.treatment, row=1, replacement="Drug_A")

# Behavior:
cf_vec[1]  # → "Drug_A" (counterfactual - substituted value)
cf_vec[2]  # → "Drug_A" (original value)
cf_vec[3]  # → "Drug_B" (original value)
cf_vec[4]  # → "Control" (original value)
```

### Implementation

All `CounterfactualVector` types implement this interface:

```julia
abstract type CounterfactualVector{T} <: AbstractVector{T} end

# Generic getindex implementation
@inline Base.getindex(v::CounterfactualVector, i::Int) =
    (i == v.row ? v.replacement : v.base[i])
```

**Key insight**: This is just a conditional branch—no array copying, no allocations.

### Typed Variants

Different data types have specialized implementations:

```julia
# For numeric variables (Float64, Int64, etc.)
mutable struct NumericCounterfactualVector{T<:Real} <: CounterfactualVector{T}
    base::Vector{T}           # Original data
    row::Int                  # Row index to override
    replacement::T            # Replacement value
end

# For categorical variables
mutable struct CategoricalCounterfactualVector{T,R} <: CounterfactualVector{CategoricalValue{T,R}}
    base::CategoricalArray{T,1,R}
    row::Int
    replacement::CategoricalValue{T,R}
end

# For boolean variables
mutable struct BoolCounterfactualVector <: CounterfactualVector{Bool}
    base::Vector{Bool}
    row::Int
    replacement::Bool
end
```

**Type stability**: Each variant has concrete types, enabling compiler optimizations.

### Mutable Updates

The vectors are **mutable** so fields can be updated without allocations:

```julia
# Update which row to override
cf_vec.row = 5

# Update the replacement value
cf_vec.replacement = "Drug_B"

# No allocations - just field assignments
```

This enables reusing the same `CounterfactualVector` for multiple contrasts.

---

## ContrastEvaluator Initialization

### Construction Process

When you create a `ContrastEvaluator`, it builds the complete counterfactual infrastructure:

```julia
evaluator = contrastevaluator(compiled, data, [:treatment, :education, :female])
```

#### Step 1: Build Counterfactual Data Structure

```julia
# Line 103 in contrast_evaluator.jl
data_counterfactual, counterfactuals = build_counterfactual_data(data, vars, 1, Float64)
```

This creates:
- **`counterfactuals`**: A tuple of typed `CounterfactualVector` objects (one per variable)
- **`data_counterfactual`**: A `NamedTuple` that merges original data with `CounterfactualVector` wrappers

**Example result**:
```julia
# Original data
data = (
    x = [1.0, 2.0, 3.0, ...],
    outcome = [0.5, 1.2, 0.8, ...],
    treatment = ["Control", "Drug_A", "Drug_B", ...],
    education = ["HS", "College", "HS", ...],
    female = [0, 1, 1, 0, ...]
)

# After build_counterfactual_data
data_counterfactual = (
    x = data.x,                                    # Original (not in vars)
    outcome = data.outcome,                        # Original (not in vars)
    treatment = counterfactuals[1],                # CategoricalCounterfactualVector
    education = counterfactuals[2],                # CategoricalCounterfactualVector
    female = counterfactuals[3]                    # NumericCounterfactualVector{Float64}
)
```

**Key insight**: Variables not in `vars` use original columns; variables in `vars` are wrapped in `CounterfactualVector`s.

#### Step 2: Pre-compute Categorical Level Mappings

```julia
# Lines 106-122 in contrast_evaluator.jl
categorical_level_maps = Dict{Symbol, Dict{String, CategoricalValue}}()

for (i, var) in enumerate(vars)
    cf_vec = counterfactuals[i]
    if cf_vec isa CategoricalCounterfactualVector
        # Build String → CategoricalValue mapping
        level_map = Dict{String, CategoricalValue}()
        base_array = cf_vec.base

        for level_str in levels(base_array)
            # Find a CategoricalValue instance for this level
            matching_idx = findfirst(x -> string(x) == level_str, base_array)
            if matching_idx !== nothing
                level_map[level_str] = base_array[matching_idx]
            end
        end

        categorical_level_maps[var] = level_map
    end
end
```

**Purpose**: Pre-computing these mappings avoids allocations during contrast evaluation. Converting strings like `"Control"` to `CategoricalValue` objects normally allocates, but looking up in a pre-built `Dict` does not.

**Example**:
```julia
categorical_level_maps[:treatment] = Dict(
    "Control" => CategoricalValue("Control", pool),
    "Drug_A" => CategoricalValue("Drug_A", pool),
    "Drug_B" => CategoricalValue("Drug_B", pool)
)
```

#### Step 3: Detect Binary Variables

```julia
# Lines 124-141 in contrast_evaluator.jl
binary_vars = Set{Symbol}()
binary_coef_indices = Dict{Symbol, Int}()

for (i, var) in enumerate(vars)
    cf_vec = counterfactuals[i]
    col = getproperty(data, var)

    if _is_truly_binary_variable(cf_vec, col)
        binary_vars = union(binary_vars, [var])
        # Find coefficient index for fast path optimization
        coef_idx = _find_binary_coefficient_index(compiled, var)
        if coef_idx !== nothing
            binary_coef_indices[var] = coef_idx
        end
    end
end
```

**Purpose**: Binary variables (0/1, true/false) have a fast path—the contrast is simply ±1 at the coefficient position, skipping full model evaluation.

#### Step 4: Allocate Buffers

```julia
# Lines 143-157 in contrast_evaluator.jl
ContrastEvaluator(
    compiled,
    vars,
    data_counterfactual,
    counterfactuals,
    Vector{Float64}(undef, length(compiled)),    # y_from_buf
    Vector{Float64}(undef, length(compiled)),    # y_to_buf
    categorical_level_maps,
    binary_vars,
    binary_coef_indices,
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
1. Update counterfactual to "from" level → evaluate → store result
2. Update counterfactual to "to" level → evaluate → store result
3. Compute difference

Let's trace through an example step-by-step.

### Example Call

```julia
contrast_modelrow!(Δ, evaluator, row=1, :treatment, "Control", "Drug")
```

This computes: `Δ = X(treatment="Drug") - X(treatment="Control")` for row 1.

### Step 1: Update Counterfactual to "from" Level

```julia
# Line 213 in contrast_evaluator.jl
update_counterfactual_for_var!(
    evaluator.counterfactuals,           # Tuple of CounterfactualVectors
    evaluator.vars,                      # [:treatment, :education, :female]
    :treatment,                          # Variable to modify
    row,                                 # Row index = 1
    "Control",                           # Baseline level
    evaluator.categorical_level_maps     # Pre-computed mappings
)
```

**What `update_counterfactual_for_var!` does** (lines 385-400 in typed_overrides.jl):

```julia
# 1. Find the CounterfactualVector for :treatment
cf_vec = get_counterfactual_for_var(counterfactuals, vars, :treatment)
# → Returns counterfactuals[1] (CategoricalCounterfactualVector for :treatment)

# 2. Update which row it's overriding
update_counterfactual_row!(cf_vec, row)
# → Sets cf_vec.row = 1

# 3. Use pre-computed mapping to get CategoricalValue (zero allocations!)
level_map = categorical_level_maps[:treatment]
replacement_str = string("Control")  # → "Control"
cat_val = level_map[replacement_str]  # Dict lookup - no allocation!
# → Gets CategoricalValue("Control") from pre-built map

# 4. Update the replacement value (mutable field assignment)
update_counterfactual_replacement!(cf_vec, cat_val)
# → Sets cf_vec.replacement = CategoricalValue("Control")
```

**Result**: The `CounterfactualVector` for `:treatment` now behaves as:
```julia
data_counterfactual.treatment[1]  # → "Control" (counterfactual)
data_counterfactual.treatment[2]  # → "Drug_A" (original)
data_counterfactual.treatment[3]  # → "Drug_B" (original)
# ... all other rows return original values
```

### Step 2: Evaluate Formula with "from" Level

```julia
# Line 214 in contrast_evaluator.jl
evaluator.compiled(evaluator.y_from_buf, evaluator.data_counterfactual, row)
```

**What happens**:
1. The compiled formula evaluates row 1 using `data_counterfactual`
2. When it accesses `data_counterfactual.treatment[1]`, it gets `"Control"` (the counterfactual)
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

```julia
# Line 216 in contrast_evaluator.jl
update_counterfactual_for_var!(
    evaluator.counterfactuals,
    evaluator.vars,
    :treatment,
    row,
    "Drug",  # Now set to comparison level
    evaluator.categorical_level_maps
)
```

**Same process as Step 1**, but now:
```julia
cf_vec.row = 1                                    # Same row
cf_vec.replacement = CategoricalValue("Drug")     # Different level
```

**Result**: The same `CounterfactualVector` now returns `"Drug"` for row 1.

### Step 4: Evaluate Formula with "to" Level

```julia
# Line 217 in contrast_evaluator.jl
evaluator.compiled(evaluator.y_to_buf, evaluator.data_counterfactual, row)
```

**Result**: `y_to_buf` contains the model matrix row X₁ with `treatment="Drug"`

### Step 5: Compute Discrete Effect

```julia
# Line 219 in contrast_evaluator.jl
Δ .= evaluator.y_to_buf .- evaluator.y_from_buf
```

**Result**: `Δ = X₁ - X₀` (the discrete effect vector)

This is the **contrast vector** showing how each coefficient contributes to the treatment effect.

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Update CounterfactualVector: treatment[1] → "Control"       │
│    - Mutable field assignment: cf_vec.replacement = "Control"   │
│    - Uses pre-computed categorical mapping (0 allocations)      │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Evaluate formula with data_counterfactual                    │
│    - compiled(y_from_buf, data_counterfactual, row=1)          │
│    - Formula accesses data_counterfactual.treatment[1]          │
│    - CounterfactualVector returns "Control" (not original)      │
│    - Result: y_from_buf = X₀ (model row with Control)          │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Update CounterfactualVector: treatment[1] → "Drug"          │
│    - Mutable field assignment: cf_vec.replacement = "Drug"      │
│    - Same vector, just update replacement field (0 allocations) │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Evaluate formula with data_counterfactual (again)            │
│    - compiled(y_to_buf, data_counterfactual, row=1)            │
│    - CounterfactualVector now returns "Drug"                    │
│    - Result: y_to_buf = X₁ (model row with Drug)               │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Compute difference                                           │
│    - Δ .= y_to_buf .- y_from_buf                               │
│    - Result: Δ = X₁ - X₀ (discrete effect)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Gradient Computation

### Purpose

Parameter gradients enable **uncertainty quantification** via the delta method:

```julia
# Compute gradient ∇β where ∇β[i] = ∂(discrete_effect)/∂β[i]
contrast_gradient!(∇β, evaluator, row, :treatment, "Control", "Drug", β)

# Delta method standard error
se = sqrt(∇β' * vcov_matrix * ∇β)
```

### Linear Scale Gradients

For discrete effects on the linear predictor scale η = Xβ:

```julia
discrete_effect = η₁ - η₀ = (X₁'β) - (X₀'β) = (X₁ - X₀)'β
```

The gradient is simply:
```julia
∇β = ΔX = X₁ - X₀
```

**Implementation** (lines 786-796 in contrast_evaluator.jl):
```julia
# Update to "from" level and evaluate
update_counterfactual_for_var!(evaluator.counterfactuals, evaluator.vars, var, row, from, ...)
evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)
# → xrow_from_buf = X₀

# Update to "to" level and evaluate
update_counterfactual_for_var!(evaluator.counterfactuals, evaluator.vars, var, row, to, ...)
evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)
# → xrow_to_buf = X₁

# Gradient is the contrast vector
∇β .= xrow_to_buf .- xrow_from_buf  # ∇β = X₁ - X₀
```

### Response Scale Gradients

For discrete effects on the response scale μ = g⁻¹(η):

```julia
discrete_effect = μ₁ - μ₀ = g⁻¹(η₁) - g⁻¹(η₀)
```

By the chain rule:
```julia
∇β = g'(η₁) × X₁ - g'(η₀) × X₀
```

Where `g'(η) = dμ/dη` is the link function derivative.

**Implementation** (lines 825-842 in contrast_evaluator.jl):
```julia
# Compute X₀ and η₀ = X₀'β
update_counterfactual_for_var!(...)
evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)
η₀ = dot(β, evaluator.xrow_from_buf)

# Compute X₁ and η₁ = X₁'β
update_counterfactual_for_var!(...)
evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)
η₁ = dot(β, evaluator.xrow_to_buf)

# Compute link function derivatives
g_prime_η₀ = _dmu_deta(link, η₀)  # dμ/dη at η₀
g_prime_η₁ = _dmu_deta(link, η₁)  # dμ/dη at η₁

# Apply chain rule
∇β .= g_prime_η₁ .* xrow_to_buf .- g_prime_η₀ .* xrow_from_buf
```

### Supported Link Functions

All GLM link functions are supported:
- `IdentityLink`: g'(η) = 1
- `LogLink`: g'(η) = exp(η)
- `LogitLink`: g'(η) = exp(η) / (1 + exp(η))²
- `ProbitLink`: g'(η) = φ(η) (standard normal PDF)
- `CloglogLink`, `CauchitLink`, `InverseLink`, `SqrtLink`, etc.

---

## Performance Characteristics

### Zero Allocations Achieved Through

1. **Pre-allocated buffers**
   - All output buffers allocated once during construction
   - Reused across all contrast computations
   - Buffers: `y_from_buf`, `y_to_buf`, `xrow_from_buf`, `xrow_to_buf`, `gradient_buffer`

2. **Mutable CounterfactualVectors**
   - Update fields in-place: `cf_vec.row = new_row`, `cf_vec.replacement = new_value`
   - No array copying or temporary allocations
   - Type-stable concrete types enable compiler optimizations

3. **Pre-computed categorical mappings**
   - String → CategoricalValue lookups cached at construction
   - Dictionary lookups don't allocate (just pointer access)
   - Avoids repeated level searches in CategoricalArray

4. **Type specialization**
   - All CounterfactualVector types are concrete (not abstract)
   - Compiled formula is fully type-specialized
   - No runtime dispatch on hot paths

### Memory Efficiency Comparison

**Traditional approach** (allocates O(n) memory):
```julia
# Copy entire dataset and modify
data_control = deepcopy(data)
data_control.treatment .= "Control"  # Allocate new column!
X₀ = modelmatrix(formula, DataFrame(data_control))

data_drug = deepcopy(data)
data_drug.treatment .= "Drug"  # Another allocation!
X₁ = modelmatrix(formula, DataFrame(data_drug))

Δ = X₁[row, :] - X₀[row, :]
```

**Memory used**: ~2n + O(model_matrix_size)

**CounterfactualVector approach** (O(1) memory):
```julia
# Create evaluator once (one-time cost)
evaluator = contrastevaluator(compiled, data, [:treatment])
Δ = Vector{Float64}(undef, length(compiled))

# Compute contrast (zero allocations)
contrast_modelrow!(Δ, evaluator, row, :treatment, "Control", "Drug")
```

**Memory used**: ~32 bytes (for mutable field updates)

**Savings**: >99.999% memory reduction for large datasets

### Timing Benchmarks

From test suite (`test/test_contrast_evaluator.jl`):

```julia
# Construction (one-time cost)
@benchmark contrastevaluator($compiled, $data, [:treatment, :education])
# Typical: ~10-50μs, <50KB allocated

# Contrast computation (repeated operations)
@benchmark contrast_modelrow!($Δ, $evaluator, 1, :treatment, "Control", "Drug")
# Typical: ~50-200ns, 0 bytes allocated

# Gradient computation
@benchmark contrast_gradient!($∇β, $evaluator, 1, :treatment, "Control", "Drug", $β)
# Typical: ~100-300ns, 0 bytes allocated
```

### Batch Processing

Processing multiple contrasts reuses the same buffers:

```julia
evaluator = contrastevaluator(compiled, data, [:treatment])
Δ = Vector{Float64}(undef, length(compiled))

# Zero allocations for all iterations
for row in 1:1000
    contrast_modelrow!(Δ, evaluator, row, :treatment, "Control", "Drug")
    # Process Δ...
end
```

**Performance**: Constant memory usage regardless of number of contrasts.

---

## Advanced Topics

### Binary Variable Fast Path

Binary variables (0/1, true/false) have optimized computation that skips formula evaluation:

```julia
# For binary variables, the contrast is simply ±1 at the coefficient position
function _contrast_modelrow_binary_fast_path!(Δ, evaluator, var, from, to)
    coef_idx = evaluator.binary_coef_indices[var]

    # Zero out all coefficients
    @inbounds for i in eachindex(Δ)
        Δ[i] = 0.0
    end

    # Set single non-zero element
    contrast_direction = _binary_contrast_direction(from, to)  # ±1.0
    @inbounds Δ[coef_idx] = contrast_direction

    return Δ
end
```

**Speedup**: ~10x faster than general path for binary variables.

### Multiple Variables

The same evaluator can process different variables:

```julia
evaluator = contrastevaluator(compiled, data, [:treatment, :region, :education])
Δ_treatment = Vector{Float64}(undef, length(compiled))
Δ_region = Vector{Float64}(undef, length(compiled))

# Different variables, same evaluator, same buffers
contrast_modelrow!(Δ_treatment, evaluator, 1, :treatment, "Control", "Drug")
contrast_modelrow!(Δ_region, evaluator, 1, :region, "North", "South")
```

**Memory**: O(1) regardless of number of variables or contrasts.

### Integration with Compiled Formula

The compiled formula is **completely unaware** that it's using CounterfactualVectors:

```julia
# Inside the compiled formula evaluation
function (compiled::UnifiedCompiled)(output, data, row)
    # Access column (might be CounterfactualVector)
    treatment_val = data.treatment[row]  # Dispatches to getindex

    # CounterfactualVector returns replacement value if row matches
    # Otherwise returns original value
    # Formula sees no difference!
end
```

**Transparency**: The `AbstractVector` interface makes CounterfactualVectors indistinguishable from regular vectors to the formula compiler.

### Type Stability Validation

All CounterfactualVector types maintain concrete element types:

```julia
# Numeric: Float64 in, Float64 out
cf_numeric::NumericCounterfactualVector{Float64}
eltype(cf_numeric) === Float64  # ✓

# Categorical: CategoricalValue in, CategoricalValue out
cf_cat::CategoricalCounterfactualVector{String, UInt32}
eltype(cf_cat) === CategoricalValue{String, UInt32}  # ✓

# Boolean: Bool in, Bool out
cf_bool::BoolCounterfactualVector
eltype(cf_bool) === Bool  # ✓
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

All with **zero allocations** after the one-time construction cost, enabling efficient batch processing of thousands of contrasts without memory overhead.