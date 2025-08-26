# Derivative System UI Simplification Analysis

## Current API Assessment

The current derivative system has these separate functions:
- `build_derivative_evaluator()` - creates evaluator object
- `derivative_modelrow!()` / `derivative_modelrow()` - ForwardDiff derivatives  
- `derivative_modelrow_fd!()` / `derivative_modelrow_fd()` - finite difference fallback
- `contrast_modelrow!()` / `contrast_modelrow()` - discrete contrasts
- `marginal_effects_eta!()` / `marginal_effects_eta()` - marginal effects on η
- `marginal_effects_mu!()` / `marginal_effects_mu()` - marginal effects on μ

**Assessment**: This API is actually **well-designed** with clear purpose-built functions and explicit performance control.

## Proposed Unified Interface

Extend the existing `modelrow!` / `modelrow` pattern with multiple dispatch:

```julia
# Core evaluation (existing)
modelrow!(output, compiled, data, row)
modelrow(compiled, data, row)

# Derivatives - return Jacobian ∂X/∂vars
modelrow!(J, compiled, data, row, ::Derivatives{vars})  
modelrow(compiled, data, row, ::Derivatives{vars})

# Finite difference derivatives  
modelrow!(J, compiled, data, row, ::FiniteDifferences{vars})
modelrow(compiled, data, row, ::FiniteDifferences{vars})

# Contrasts - discrete changes
modelrow!(Δ, compiled, data, row, ::Contrast{var, from, to})
modelrow(compiled, data, row, ::Contrast{var, from, to}) 

# Marginal effects on η = Xβ
modelrow!(g, compiled, data, row, ::MarginalEffectsEta{vars}, β)
modelrow(compiled, data, row, ::MarginalEffectsEta{vars}, β)

# Marginal effects on μ = g⁻¹(η) 
modelrow!(g, compiled, data, row, ::MarginalEffectsMu{vars}, β, link)
modelrow(compiled, data, row, ::MarginalEffectsMu{vars}, β, link)
```

## Dispatch Types

```julia
struct Derivatives{Vars} end
struct FiniteDifferences{Vars} 
    step::Union{Symbol, Float64}
end
struct Contrast{Var, From, To} end
struct MarginalEffectsEta{Vars} end  
struct MarginalEffectsMu{Vars} end

# Constructor functions
Derivatives(vars::Vector{Symbol}) = Derivatives{Tuple(vars)}()
FiniteDifferences(vars::Vector{Symbol}; step=:auto) = FiniteDifferences{Tuple(vars)}(step)
Contrast(var::Symbol, from, to) = Contrast{var, from, to}()
MarginalEffectsEta(vars::Vector{Symbol}) = MarginalEffectsEta{Tuple(vars)}()
MarginalEffectsMu(vars::Vector{Symbol}) = MarginalEffectsMu{Tuple(vars)}()
```

## Implementation Strategy

### Benefits
1. **Unified API**: All evaluation through same `modelrow!/modelrow` functions
2. **Type-based dispatch**: Clear, compile-time specialization  
3. **Backward compatibility**: Existing `modelrow!` calls unchanged
4. **Performance**: Can maintain zero-allocation paths
5. **Discoverability**: Single function name to remember

### Implementation Approach
1. **Create dispatch types** in `derivatives.jl`
2. **Add new method signatures** alongside existing ones
3. **Internal state management**: Handle `DerivativeEvaluator` creation automatically
4. **Preserve performance**: Keep zero-allocation execution paths
5. **Migration path**: Deprecate old API gradually

### Key Design Decisions
- **Automatic evaluator creation**: No need for explicit `build_derivative_evaluator`
- **Type-based variable specification**: `Derivatives{(:x, :z)}()` vs `vars=[:x, :z]`
- **Consistent output orientation**: Maintain `(n_terms, n_vars)` for Jacobians
- **Link specification**: Part of dispatch type for marginal effects

## Migration Examples

### Current API
```julia
# Derivatives
de = build_derivative_evaluator(compiled, data; vars=[:x, :z])
J = derivative_modelrow(de, 1)

# Finite differences
J_fd = derivative_modelrow_fd(compiled, data, 1; vars=[:x, :z])

# Contrasts
Δ = contrast_modelrow(compiled, data, 1; var=:group, from="A", to="B")

# Marginal effects
gη = marginal_effects_eta(de, β, 1)
gμ = marginal_effects_mu(de, β, 1; link=LogitLink())
```

### New Unified API
```julia
# Derivatives
J = modelrow(compiled, data, 1, Derivatives([:x, :z]))

# Finite differences  
J_fd = modelrow(compiled, data, 1, FiniteDifferences([:x, :z]))

# Contrasts
Δ = modelrow(compiled, data, 1, Contrast(:group, "A", "B"))

# Marginal effects
gη = modelrow(compiled, data, 1, MarginalEffectsEta([:x, :z]), β)
gμ = modelrow(compiled, data, 1, MarginalEffectsMu([:x, :z]), β, LogitLink())
```

## Performance Considerations

- **Zero-allocation paths**: Maintain existing performance characteristics
- **Automatic caching**: Internal `DerivativeEvaluator` management with caching
- **Type specialization**: Compile-time dispatch for optimal performance
- **Memory efficiency**: Reuse buffers across calls when possible

## Critical Analysis: Is This Worth It?

### Benefits of Unified Interface
- **Single entry point**: `modelrow!` for everything
- **Familiar pattern**: Users already know this interface  
- **Type safety**: Compile-time dispatch

### Significant Costs
- **Implementation complexity**: Need internal evaluator caching to maintain performance
- **Performance risk**: Could lose zero-allocation guarantees without careful implementation
- **API confusion**: Multiple overloaded signatures for same function name
- **Loss of explicit control**: Current API gives users direct control over performance-critical objects

### Current API Strengths
The existing API is actually **excellent design**:

```julia
# Clear, purpose-built functions with explicit performance control
de = build_derivative_evaluator(compiled, data; vars=[:x, :z])  # Build once, control lifecycle
J = derivative_modelrow!(buffer, de, row)                       # Reuse many times, zero allocations
```

**Key strengths**:
- **Zero ambiguity**: Each function has one clear purpose
- **Explicit performance control**: Users manage evaluator lifecycle  
- **Proven robustness**: 2005+ tests, consistent zero-allocation execution
- **Consistent patterns**: All follow same `!` vs non-`!` convention

## **Recommendation: Keep Current API**

The "complexity" being solved isn't really complexity - it's **explicit performance control** that advanced users of a performance-critical library actually need and want.

### Better Approach: Add Convenience Wrappers

Instead of a major API overhaul, add simple convenience functions:

```julia
# Keep existing high-performance API as-is
de = build_derivative_evaluator(compiled, data; vars=[:x, :z])
J = derivative_modelrow!(buffer, de, row)  # Zero allocations

# Add convenience wrappers for casual use
function derivatives(compiled, data, row; vars)
    de = build_derivative_evaluator(compiled, data; vars)
    return derivative_modelrow(de, row)  # Allocates, but simple
end
```

This provides **both** simplicity for casual users and performance control for demanding applications.

## **Conclusion**

The multiple dispatch approach would trade **explicit performance control** for **API uniformity** - not a good tradeoff for a performance-critical statistical computing library.

**Final verdict: Keep the current well-designed, performant API.**