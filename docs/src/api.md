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

Dual-backend derivatives with preallocated buffers for efficiency.

### Evaluator Construction

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

### Link Function Derivatives

```@docs
_dmu_deta
_d2mu_deta2
```

## Categorical Contrasts

```@docs
ContrastEvaluator
contrastevaluator
CategoricalLevelMap
contrast_modelrow!
contrast_gradient!
contrast_gradient
supported_link_functions
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