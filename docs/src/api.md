# API Reference

API reference for FormulaCompiler.jl functions and types.

## Core Compilation Functions

```@docs
compile_formula
get_or_compile_formula
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