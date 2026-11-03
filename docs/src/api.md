# API Reference

API reference for FormulaCompiler.jl functions and types.

## Core Compilation Functions

```@docs
compile_formula
get_or_compile_formula
with_private_scratch
extract_variable_names
fixed_effects_form
```

### Compiled-Formula Cache

`get_or_compile_formula` (and `modelrow`, `modelrow!(row, model, data, i)`) keep compiled
formulas in a bounded least-recently-used cache. Each entry holds its fitted model.

```@docs
clear_compiled_cache!
set_compiled_cache_capacity!
compiled_cache_capacity
```

## Model Row Evaluation

```@docs
modelrow
modelrow!
ModelRowEvaluator
```

## Derivatives

FormulaCompiler provides computational primitives for computing derivatives of model matrix rows with respect to continuous variables. These functions enable zero-allocation Jacobian computation using either automatic differentiation (ForwardDiff) or finite differences.

For marginal effects, standard errors, and complete statistical workflows, see [Margins.jl](https://github.com/emfeltham/Margins.jl).

### Evaluator Construction

**Recommended**: Use the unified dispatcher for user-facing code:

```julia
# Automatic differentiation (preferred)
de = derivativeevaluator(:ad, compiled, data, [:x, :z])

# Finite differences
de = derivativeevaluator(:fd, compiled, data, [:x, :z])
```

**Advanced**: Direct constructor functions (primarily for internal use):

```@docs
derivativeevaluator
derivativeevaluator_fd
derivativeevaluator_ad
```

### Evaluator Interface

The concrete evaluator types and their fields are internal; use these instead.

```@docs
AbstractDerivativeEvaluator
derivative_variables
derivative_backend
```

### Jacobian Computation

```@docs
derivative_modelrow!
derivative_modelrow
```

### Variable Identification

```@docs
continuous_variables
```

### Link Function Derivatives

Computational primitives for GLM link function derivatives (used by Margins.jl for computing marginal effects on the mean response).

```@docs
_dmu_deta
_d2mu_deta2
supported_link_functions
```

## Categorical Contrasts

```@docs
ContrastEvaluator
contrastevaluator
CategoricalLevelMap
contrast_modelrow!
level_modelrow!
contrast_endpoints
categorical_levels
categorical_variables
baseline_level
contrast_gradient!
contrast_gradient
```

## Categorical Mixtures

Utilities for constructing and validating categorical mixtures used in efficient profile-based marginal effects.

```@docs
mix
CategoricalMixture
MixtureWithLevels
validate_mixture_against_data
mixture_to_scenario_value
```

## Utilities

```@docs
not
```