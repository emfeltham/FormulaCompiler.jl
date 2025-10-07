# API Reference

API reference for FormulaCompiler.jl functions and types.

## Core Compilation Functions

```@docs
compile_formula
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
fd_jacobian_column!
```

### Marginal Effects

```@docs
marginal_effects_eta!
marginal_effects_mu!
```

### Utilities

```@docs
continuous_variables
delta_method_se
```

## Categorical Contrasts

```@docs
contrastevaluator
contrast_modelrow!
contrast_gradient!
```

## Categorical Mixtures

Utilities for constructing and validating categorical mixtures used in efficient profile-based marginal effects.

```@docs
mix
CategoricalMixture
MixtureWithLevels
validate_mixture_against_data
create_balanced_mixture
mixture_to_scenario_value
```