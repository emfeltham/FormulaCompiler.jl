# FormulaCompiler.jl Derivatives System Design

## Overview

The derivatives system provides high-performance automatic differentiation and finite difference computations for statistical models, achieving zero-allocation execution through extensive type specialization and preallocation strategies.

## Architecture

### Modular Organization

The derivatives system is organized into focused modules under `src/evaluation/derivatives/`:

```
src/evaluation/derivatives/
├── types.jl              # Core types and data structures
├── overrides.jl          # Override vector implementations  
├── evaluator.jl          # DerivativeEvaluator construction
├── automatic_diff.jl     # ForwardDiff implementations
├── finite_diff.jl        # Finite difference implementations
├── marginal_effects.jl   # η and μ marginal effects
├── contrasts.jl          # Discrete contrasts
├── link_functions.jl     # GLM link function derivatives
└── utilities.jl          # Helper functions
```

**Main interface:** `src/evaluation/derivatives.jl` loads all modules and exports the public API.

## Core Components

### 1. Type System (`types.jl`)

**`DerivativeEvaluator`**: Central data structure maintaining all state for zero-allocation computations
- Concrete typed closures and ForwardDiff configurations
- Preallocated buffers for Jacobians, gradients, and finite differences
- Multiple data override systems (Float64 and Dual-typed)
- Cached column references for zero-allocation access

**`DerivClosure`**: Callable closure for ForwardDiff with reusable buffers
**`GradClosure`**: Scalar gradient closure for η = Xβ computations

### 2. Override System (`overrides.jl`)

**Variable substitution for derivative computations:**
- `SingleRowOverrideVector`: General-purpose (uses `Any` eltype)
- `TypedSingleRowOverrideVector{T}`: Type-preserving version
- `FDOverrideVector`: Concrete Float64 specialization for FD computations

### 3. Evaluator Construction (`evaluator.jl`)

**`build_derivative_evaluator(compiled, data; vars, chunk)`**:
- Builds reusable evaluator with concrete typed closures/configs
- Pre-caches column references as NTuple for unrolled access
- Preallocates all computation buffers
- Zero allocations after warmup

### 4. Backend Implementations

**Automatic Differentiation (`automatic_diff.jl`)**:
- `derivative_modelrow!`: ForwardDiff Jacobian computation
- `marginal_effects_eta_grad!`: Direct gradient computation for η

**Finite Differences (`finite_diff.jl`)**:
- `derivative_modelrow_fd_pos!`: Zero-allocation FD Jacobian (positional hot path)
- Generated functions with compile-time loop unrolling
- Central difference approximations with automatic step selection

### 5. Marginal Effects (`marginal_effects.jl`)

**Backend Selection Architecture**:
```julia
marginal_effects_eta!(g, de, beta, row; backend=:ad/:fd)
marginal_effects_mu!(g, de, beta, row; link, backend=:ad/:fd)
```

**Backends:**
- `:ad`: ForwardDiff (≤288 bytes allocation, faster, more accurate)
- `:fd`: Finite differences (0 bytes allocation, slightly slower)

### 6. Link Functions (`link_functions.jl`)

**GLM link function derivatives (`dμ/dη`):**
- Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt, InverseSquare
- Optimized inline implementations with mathematical constants

### 7. Contrasts (`contrasts.jl`)

**Discrete contrasts**: `contrast_modelrow!` for categorical variable differences
- Handles CategoricalArrays with proper level consistency
- Row-local override system for variable substitution

### 8. Utilities (`utilities.jl`)

**Helper functions:**
- `continuous_variables`: Discover Real-typed variables in compiled formulas
- Excludes categorical variables detected via ContrastOps

## Performance Characteristics

### Allocation Profiles (Benchmark Results)

| Path | Backend | Memory (bytes) | Time (ns) |
|------|---------|----------------|-----------|
| Core compilation | - | 0 | 7 |
| FD Jacobian (evaluator) | `:fd` | 0 | 45 |
| AD Jacobian | `:ad` | 256 | 424 |
| η marginal effects | `:fd` | 0 | 58 |
| η marginal effects | `:ad` | 288 | 497 |
| μ marginal effects (Logit) | `:fd` | 0 | 80 |
| μ marginal effects (Logit) | `:ad` | 256 | 482 |

### Key Performance Features

1. **Zero-allocation FD backend**: Achieved through:
   - Concrete `FDOverrideVector` types
   - Pre-cached column references (`fd_columns`)
   - Generated functions with compile-time loop unrolling
   - All buffers preallocated in evaluator

2. **Type specialization**: All positions and operations embedded in type parameters

3. **Buffer reuse**: Extensive preallocation and reuse across calls

4. **Validation**: Cross-validated between AD and FD methods (rtol=1e-6, atol=1e-8)

## Usage Patterns

### Basic Derivative Computation
```julia
# Build evaluator (once per model + variable set)
de = build_derivative_evaluator(compiled, data; vars=[:x, :z])

# Zero-allocation Jacobian
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow_fd_pos!(J, de, row)  # 0 bytes

# Marginal effects with backend selection
g = Vector{Float64}(undef, length(vars))
marginal_effects_eta!(g, de, β, row; backend=:fd)  # 0 bytes
marginal_effects_mu!(g, de, β, row; link=LogitLink(), backend=:fd)  # 0 bytes
```

### Performance-Critical Applications
- Use `:fd` backend for strict zero-allocation requirements
- Use `:ad` backend for speed and numerical accuracy (default)
- Compile evaluator once, reuse across many rows

## Design Principles

1. **Position-mapped compilation**: All complexity resolved at compile time
2. **Type stability**: Zero runtime dispatch through concrete type parameters  
3. **Memory efficiency**: Pre-allocation with exact buffer sizing
4. **Modular architecture**: Focused responsibilities, clear interfaces
5. **Backend flexibility**: User choice between performance and accuracy trade-offs
6. **Validation**: Robust cross-checking between methods

## Future Improvements

From TODO comments in code:
1. **Concrete types**: Eliminate remaining `Any`-typed fields
2. **Single-column API**: Per-variable derivative computation
3. **Enhanced η-gradient path**: Further allocation reductions
4. **Delta-method standard errors**: Variance computation integration

## Integration

The derivatives system integrates seamlessly with:
- **GLM.jl**: Linear and generalized linear models
- **MixedModels.jl**: Mixed-effects models (fixed effects extraction)  
- **CategoricalArrays.jl**: All contrast types supported
- **Tables.jl**: Universal table format support
- **StatsModels.jl**: Complete formula system compatibility

This design provides a production-ready, high-performance derivative computation system suitable for both research and production statistical computing applications.