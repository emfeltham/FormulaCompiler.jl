# Scenario-Based Derivative Computation

## Overview

When computing derivatives in counterfactual scenarios with categorical overrides, use standalone derivative functions rather than cached `DerivativeEvaluator` objects to maintain both statistical correctness and performance.

## The Pattern

### Recommended: Standalone Functions with Scenarios

```julia
using FormulaCompiler, GLM, Tables

# Setup
model = lm(@formula(y ~ x * group + z), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)
continuous_vars = [:x, :z]

# Create scenarios for profile analysis
scenarios = create_scenario_grid("analysis", data, Dict(
    :group => ["A", "B", "C"],
    :z => [1.0, 2.0, 3.0]
))

# Compute derivatives in each scenario context
results = []
for scenario in scenarios
    # Use standalone functions with scenario data
    J = Matrix{Float64}(undef, length(compiled), length(continuous_vars))
    derivative_modelrow_fd!(J, compiled, scenario.data, 1; vars=continuous_vars)
    
    # Convert to marginal effects on η
    β = coef(model)
    marginal_effects_eta = J' * β
    
    # For marginal effects on μ, apply link function derivative
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, scenario.data, 1)
    η = dot(β, row_vec)
    dmu_deta = compute_link_derivative(link, η)
    marginal_effects_mu = marginal_effects_eta .* dmu_deta
    
    push!(results, (scenario=scenario, eta=marginal_effects_eta, mu=marginal_effects_mu))
end
```

### Problematic: Cached Evaluator with Scenarios

```julia
# DON'T DO THIS: Type mismatch between evaluator and scenario data
de = build_derivative_evaluator(compiled, original_data; vars=continuous_vars)
scenario = create_scenario("profile", data; group="B")
marginal_effects_eta!(g, de, β, 1)  # ERROR: de.base_data incompatible with scenario.data
```

## Why This Matters Statistically

Continuous variable derivatives **depend on categorical context** when interaction terms are present:

```julia
# Model: y ~ x * group
# Marginal effect: ∂μ/∂x = β_x + β_interaction * I(group="B")
#                                ↑
#                            Depends on categorical value!
```

Computing `∂μ/∂x` at `group="A"` vs `group="B"` produces different results, making scenario context essential for statistical correctness.

## Performance Characteristics

| Approach | Memory | Time | Statistical Correctness |
|----------|--------|------|------------------------|
| Standalone + scenarios | ~0 bytes | ~0.1ms/profile | Correct |
| Recompilation | ~1MB | ~1ms/profile | Correct |
| Cached evaluator | Type errors | N/A | Broken |

The standalone approach achieves both performance and correctness goals.

## Helper Function Pattern

For repeated scenario-based derivative computation, consider this helper:

```julia
function scenario_marginal_effects!(
    g_eta::Vector{Float64}, 
    g_mu::Vector{Float64},
    compiled::UnifiedCompiled,
    scenario_data::NamedTuple,
    β::Vector{Float64},
    row::Int,
    link::Link,
    continuous_vars::Vector{Symbol}
)
    # Compute Jacobian in scenario context
    J = Matrix{Float64}(undef, length(compiled), length(continuous_vars))
    derivative_modelrow_fd!(J, compiled, scenario_data, row; vars=continuous_vars)
    
    # η marginal effects: ∂η/∂x
    mul!(g_eta, J', β)
    
    # μ marginal effects: ∂μ/∂x = (∂μ/∂η) * (∂η/∂x)
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, scenario_data, row)
    η = dot(β, row_vec)
    dmu_deta = GLM.mueta(link, η)  # Link function derivative
    @. g_mu = g_eta * dmu_deta
    
    return g_eta, g_mu
end
```

## Performance Notes

- **Zero allocations**: Preallocate `J`, `g_eta`, `g_mu`, `row_vec` outside loops
- **Backend selection**: Use `backend=:fd` for zero allocations or `backend=:ad` for speed
- **Memory efficiency**: Scenario system provides O(1) memory overhead regardless of data size

This pattern enables efficient, statistically correct derivative computation across counterfactual scenarios without recompilation overhead.