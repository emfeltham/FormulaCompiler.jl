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
- **Scenario analysis**: Memory-efficient variable overrides for counterfactuals
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

# Scenario analysis with overrides
scenario = create_scenario("policy", data; x = 2.0, group = "A")
compiled(row_vec, scenario.data, 1)  # Evaluate with overrides
```

## Supported Formulas

- Basic terms: `x`, `log(z)`, `x^2`
- Categorical variables with all contrast types
- Interactions: `x * group`, `x * y * z`
- Functions: `log`, `exp`, `sqrt`, `sin`, `cos`, `abs`, `^`
- Complex formulas: `x * log(z) * group + sqrt(abs(y))`

## Performance

- Single row: ~50ns, 0 allocations
- 10-100x faster than `modelmatrix()[row, :]`
- Memory efficient: O(1) for scenarios vs O(n) for data copies
"""
module FormulaCompiler

################################ Dependencies ################################

# Development dependencies (kept out of production builds)

# Core dependencies
using Dates: now
using Statistics
using StatsModels, GLM, CategoricalArrays, Tables, DataFrames
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
export not, OverrideVector

# Mixture system
include("mixtures/types.jl")
include("mixtures/constructors.jl") 
include("mixtures/validation.jl")
export CategoricalMixture, MixtureWithLevels
export mix
export validate_mixture_against_data, create_balanced_mixture, mixture_to_scenario_value

################################# Integration #################################

# External package integration
include("integration/mixed_models.jl")

################################# Compilation #################################

# Compilation system (unified)
include("compilation/compilation.jl")
export compile_formula

################################## Scenarios ##################################

# Override and scenario system (needed by modelrow)
include("scenarios/overrides.jl")
export create_categorical_override, create_scenario_grid
export DataScenario, create_scenario, create_override_data, create_override_vector

################################# Evaluation #################################

# High-level evaluation interface
include("evaluation/modelrow.jl")
export ModelRowEvaluator, modelrow!, modelrow

################################ Derivatives ################################

# ForwardDiff-based derivative evaluation (zero-alloc after warmup)
include("evaluation/derivatives.jl")
export build_derivative_evaluator, derivative_modelrow!, derivative_modelrow
export derivative_modelrow_fd!, derivative_modelrow_fd
export derivative_modelrow_fd_pos!
export contrast_modelrow!, contrast_modelrow
export continuous_variables
export marginal_effects_eta_ad!, marginal_effects_eta_fd!
export marginal_effects_mu_ad!, marginal_effects_mu_fd!
export me_mu_grad_beta!
export delta_method_se, accumulate_ame_gradient!

############################## Development Tools ##############################
# (No dev utilities included in production module.)

############################## Future Features ##############################

# Derivative system (under development)
# include("derivatives/step1_foundation.jl")
# include("derivatives/step2_functions.jl")
# export compile_derivative_formula

end # end module
