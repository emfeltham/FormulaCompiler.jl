"""
    FormulaCompiler

High-performance, zero-allocation statistical formula evaluation for Julia.

## Overview

FormulaCompiler transforms Julia statistical models into specialized, type-stable evaluators 
that achieve zero-allocation performance (~50ns per row) through compile-time specialization 
and position mapping.

## Key Features

- **Zero allocations**: Evaluates model matrices without any runtime allocations
- **Universal compatibility**: Works with any StatsModels.jl formula
- **Ecosystem integration**: Supports GLM.jl, MixedModels.jl, StandardizedPredictors.jl
- **CounterfactualVector system**: Type-stable single-row perturbations for counterfactuals
- **Type specialization**: All operations resolved at compile time

## Architecture

The package uses a **position mapping system** that converts statistical formulas into 
type-specialized operations:

1. **Formula decomposition**: Extracts terms from fitted models with applied schemas
2. **Position allocation**: Maps terms to scratch/output positions at compile time  
3. **Type specialization**: Embeds positions in operation types for zero-allocation execution

## Main API

```julia
# Compile a fitted model
compiled = compile_formula(model, data)

# Zero-allocation evaluation  
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, row_idx)  # ~50ns, 0 allocations
```

## Example Usage

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Fit a model
df = DataFrame(
    y = randn(1000),
    x = randn(1000), 
    group = rand(["A", "B", "C"], 1000)
)
model = lm(@formula(y ~ x * group + log(abs(x))), df)

# Compile for fast evaluation
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Evaluate rows with zero allocations
compiled(row_vec, data, 1)     # First row
compiled(row_vec, data, 500)   # 500th row

# CounterfactualVector functions for single-row perturbations
# Population analysis: use simple loops with existing row-wise functions
```

## Supported Formulas

- Basic terms: `x`, `log(z)`, `x^2`
- Categorical variables with all contrast types
- Interactions: `x * group`, `x * y * z`
- Functions: `log`, `exp`, `sqrt`, `sin`, `cos`, `abs`, `^`
- Complex formulas: `x * log(z) * group + sqrt(abs(y))`

"""
module FormulaCompiler

################################ Dependencies ################################

# Development dependencies (kept out of production builds)

# Core dependencies
using Dates: now
using Statistics
using StatsModels, GLM, CategoricalArrays, Tables, DataFrames
using CategoricalArrays: CategoricalValue
using LinearAlgebra: dot, I, mul!
using ForwardDiff
using Base.Iterators: product # -> compute_kronecker_pattern

# External package integration
import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel, RandomEffectsTerm
using StandardizedPredictors: ZScoredTerm

################################# Core System #################################

# Core utilities and types
include("core/utilities.jl")
export not

# Mixture system
include("mixtures/types.jl")
include("mixtures/constructors.jl") 
include("mixtures/validation.jl")
export CategoricalMixture, MixtureWithLevels
export mix
export validate_mixture_against_data, mixture_to_scenario_value
# DEPRECATED (2025-10-07): create_balanced_mixture moved to Margins.jl - will be removed in v2.0

################################# Integration #################################

# External package integration
include("integration/mixed_models.jl")

################################# Compilation #################################

# Compilation system (unified)
include("compilation/compilation.jl")
include("compilation/counterfactual_vectors.jl")
export compile_formula, get_or_compile_formula

################################## Scenarios ##################################

# Row-wise counterfactual system only (population system removed)
# Note: Population analysis should use row-wise CounterfactualVector functions in loops

################################# Evaluation #################################

# High-level evaluation interface
include("evaluation/modelrow.jl")
export ModelRowEvaluator, modelrow!, modelrow

################################ Derivatives ################################

# ForwardDiff-based derivative evaluation (zero-alloc after warmup)
include("evaluation/derivatives.jl")
export derivativeevaluator, derivativeevaluator_fd, derivativeevaluator_ad
export derivative_modelrow!, derivative_modelrow
export contrast_modelrow!
export continuous_variables
# DEPRECATED (2025-10-07): Statistical interface moved to Margins.jl - will be removed in v2.0
# export marginal_effects_eta!, marginal_effects_mu!
# export delta_method_se
# Use: `using Margins` to access these functions

export _dmu_deta, _d2mu_deta2  # Link function derivatives (computational primitives)

# Zero-allocation contrast evaluator for categorical and binary variables
include("compilation/contrast_evaluator.jl")
export ContrastEvaluator, contrastevaluator, CategoricalLevelMap
export contrast_gradient!, contrast_gradient
export supported_link_functions


end # end module
