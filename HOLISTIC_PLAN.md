# Holistic Package Design: FormulaCompiler.jl + Margins.jl

## Executive Summary

FormulaCompiler.jl has evolved into a sophisticated computational foundation with zero-allocation derivatives and modular architecture. This document outlines the strategic separation between FormulaCompiler.jl as a **computational engine** and the future Margins.jl as a **statistical interface** package.

## Current State Analysis

### FormulaCompiler.jl Architecture (Current)

```
src/
├── compilation/           # Position-mapped formula compilation
├── evaluation/           
│   ├── modelrow.jl       # Core evaluation interface
│   └── derivatives/      # Modular derivative system (9 modules)
├── scenarios/            # Override/counterfactual system
├── integration/          # GLM/MixedModels support
└── core/                 # Shared utilities
```

**Key Capabilities:**
- Zero-allocation formula evaluation (~50ns/row)
- Dual-backend derivative system (AD/FD with 0-byte FD option)
- Scenario analysis with >99% memory savings
- Complete statistical model integration

## Proposed Architecture Split

### FormulaCompiler.jl → Computational Engine

**Mission**: High-performance computational primitives for statistical computing

**Core Responsibilities:**
```julia
# Position-mapped compilation system
compiled = compile_formula(model, data)
compiled(output, data, row)  # ~50ns, 0 bytes

# Raw derivative computations
de = build_derivative_evaluator(compiled, data; vars)
derivative_modelrow_fd_pos!(J, de, row)      # 0 bytes, 44ns
marginal_effects_eta!(g, de, beta, row; backend=:fd)  # 0 bytes, 58ns
marginal_effects_mu!(g, de, beta, row; link, backend=:fd)  # 0 bytes, 79ns

# General-purpose counterfactual analysis
scenario = create_scenario("policy", data; x=2.0, group="Treatment")
grid = create_scenario_grid("sensitivity", data, param_dict)

# Mathematical utilities
continuous_variables(compiled, data)
contrast_modelrow!(Δ, compiled, data, row; var, from, to)
```

**What Stays:**
- All of `src/compilation/` (core engine)
- All of `src/evaluation/` (including modular derivatives system)
- All of `src/scenarios/` (general override system)
- Core statistical integrations (`src/integration/`)
- Performance-critical utilities (`src/core/`)

### Margins.jl → Statistical Interface

**Mission**: User-friendly marginal analysis with statistical inference

**Core Responsibilities:**
```julia
# High-level marginal effects API
results = margins(model, data; at=:means, vars=:continuous)
ame = average_marginal_effects(model, data; vars=[:x, :z])

# Statistical inference
se_results = margins_se(results; method=:delta)
test_results = margins_test(results; H0=0, vars=[:x])
ci_results = margins_ci(results; level=0.95)

# Comparative analysis
comparison = margins_comparison(model1, model2, data)
scenarios = margins_at_values(model, data; x=[1,2,3], group=["A","B"])

# Visualization and reporting
plot_margins(results; type=:effects, vars=[:x, :z])
margins_table(results; format=:latex, digits=3)
margins_summary(results; hypothesis_tests=true)
```

**What Moves to Margins.jl:**
- High-level user APIs and workflow functions
- Statistical inference (standard errors, hypothesis tests, confidence intervals)
- Visualization and reporting functions
- Documentation focused on statistical interpretation
- Convenience functions and syntactic sugar

## Benefits of This Architecture

### 1. Clear Separation of Concerns

| FormulaCompiler.jl | Margins.jl |
|-------------------|------------|
| "How to compute efficiently" | "What the results mean statistically" |
| Numerical methods | Statistical interpretation |
| Zero-allocation primitives | User-friendly workflows |
| Performance optimization | Statistical inference |

### 2. Broader Ecosystem Potential

FormulaCompiler.jl becomes a **foundation package** that other domains could build on:

- **Survival Analysis**: Hazard derivatives and counterfactual survival curves
- **Bayesian Statistics**: Posterior derivatives and sensitivity analysis  
- **Machine Learning**: Gradient computations and feature importance
- **Econometrics**: Structural model derivatives and policy simulation
- **Pharmacokinetics**: Drug concentration derivatives and dosing optimization

### 3. Focused Development Streams

**FormulaCompiler.jl Development Focus:**
- Allocation optimization and performance
- Numerical accuracy and stability
- New derivative algorithms and backends
- Integration with additional statistical packages
- Computational research and innovation

**Margins.jl Development Focus:**
- User experience and API design
- Statistical methodology and best practices
- Visualization and reporting capabilities
- Documentation and educational materials
- Applied statistical workflows

### 4. Testing and Quality Assurance

**FormulaCompiler.jl Tests:**
- Correctness (cross-validation between methods)
- Performance (allocation and timing benchmarks)
- Numerical accuracy (finite difference validation)
- Integration (compatibility with statistical packages)

**Margins.jl Tests:**
- Statistical validity (results match reference implementations)
- Edge cases (missing data, singular models, etc.)
- User workflows (end-to-end analysis pipelines)
- Visualization quality (plot correctness and aesthetics)

## Dependency Architecture

```
Margins.jl (User Interface Layer)
    ├── FormulaCompiler.jl (Computational Engine)
    ├── StatsPlots.jl (Visualization)  
    ├── StatsBase.jl (Statistical Utilities)
    ├── Bootstrap.jl (Resampling Methods)
    └── Tables.jl (Data Interface)

FormulaCompiler.jl (Foundation Layer)
    ├── ForwardDiff.jl (Automatic Differentiation)
    ├── GLM.jl/MixedModels.jl (Statistical Models)
    ├── CategoricalArrays.jl (Categorical Data)
    ├── Tables.jl (Data Interface)
    └── Base Julia (Core Language Features)
```

## Migration Strategy

### Phase 1: Stabilize FormulaCompiler.jl
1. **Complete current modularization** ✅
2. **Finalize zero-allocation derivative system** ✅  
3. **Comprehensive documentation** ✅
4. **Performance benchmarking and validation**
5. **API freeze for computational primitives**

### Phase 2: Extract Margins.jl
1. **Create new Margins.jl repository**
2. **Move high-level APIs and statistical functions**
3. **Develop user-friendly interface layer**
4. **Add statistical inference capabilities**
5. **Create visualization and reporting system**

### Phase 3: Ecosystem Integration
1. **Register both packages in Julia ecosystem**
2. **Develop tutorials and documentation**
3. **Enable extension by other statistical packages**
4. **Community feedback and iteration**

## API Design Principles

### FormulaCompiler.jl (Computational Layer)
- **Performance-first**: Every API optimized for speed and allocation
- **Type-stable**: All functions should be inferrable and concrete
- **Minimal**: Only essential computational primitives
- **Extensible**: Clear interfaces for other packages to build on

### Margins.jl (Statistical Layer)
- **User-friendly**: Intuitive APIs following statistical conventions
- **Comprehensive**: Complete workflows from model to interpretation
- **Flexible**: Multiple approaches and customization options
- **Educational**: Clear documentation with statistical context

## Success Metrics

### FormulaCompiler.jl
- **Performance**: Maintain <100ns evaluation, 0-byte derivative operations
- **Adoption**: Used as foundation by ≥3 other statistical packages
- **Stability**: API breaking changes <1 per year after v1.0
- **Quality**: >95% test coverage, comprehensive benchmarking

### Margins.jl  
- **Usability**: Complete analysis workflows in <10 lines of code
- **Statistical Validity**: Results match established packages (Stata, R)
- **Documentation**: Comprehensive tutorials and statistical guidance
- **Community**: Active user base and contributor community

## Long-term Vision

**FormulaCompiler.jl** becomes the **BLAS of statistical computing** - a high-performance foundation that other packages build upon, focusing purely on computational excellence.

**Margins.jl** becomes the **ggplot2 of marginal analysis** - an intuitive, powerful interface that makes sophisticated statistical analysis accessible to practitioners.

Together, they demonstrate how **computational foundations** and **statistical interfaces** can be cleanly separated to create more maintainable, extensible, and specialized statistical software.

## Code Migration Reality Check

### What Actually Moves from FormulaCompiler.jl to Margins.jl

**The short answer: Almost nothing.**

Looking at the current FormulaCompiler.jl codebase, essentially **all existing code stays** because it consists of computational primitives, not high-level statistical interfaces.

### Current Functions Are Mathematical Primitives (Stay in FormulaCompiler.jl)

```julia
# These STAY - they're mathematical operations, not statistical workflows
marginal_effects_eta!(g, de, beta, row; backend=:fd)      # ∂η/∂x computation
marginal_effects_mu!(g, de, beta, row; link, backend=:fd) # ∂μ/∂x with chain rule  
derivative_modelrow!(J, de, row)                          # Raw Jacobian computation
contrast_modelrow!(Δ, compiled, data, row; var, from, to) # Single contrast computation
continuous_variables(compiled, data)                      # Variable discovery
build_derivative_evaluator(compiled, data; vars)         # Evaluator construction
```

These functions perform **mathematical computations** - they compute derivatives, apply chain rules, and evaluate contrasts. They are **not** statistical workflows or user interfaces.

### What Margins.jl Builds From Scratch

Margins.jl creates **entirely new functions** that orchestrate FormulaCompiler.jl primitives into statistical workflows:

```julia
# NEW functions built on FormulaCompiler primitives
function margins(model, data; at=:means, vars=:continuous, backend=:ad)
    # Uses FormulaCompiler computational primitives:
    compiled = FormulaCompiler.compile_formula(model, data)
    if vars == :continuous
        vars = FormulaCompiler.continuous_variables(compiled, data)  # FC primitive
    end
    de = FormulaCompiler.build_derivative_evaluator(compiled, data; vars)  # FC primitive
    
    # NEW statistical logic for "at means", "at representative values", etc:
    at_values = compute_at_values(data, at)
    results = compute_marginal_effects_at_values(de, model, data, at_values, backend)
    return MarginsResult(results, model, data, vars)  # NEW statistical container type
end

function average_marginal_effects(model, data; vars=:continuous, backend=:ad)
    # Uses FC primitives to compute ME at every observation, then averages
    compiled = FormulaCompiler.compile_formula(model, data)
    de = FormulaCompiler.build_derivative_evaluator(compiled, data; vars)
    
    n_obs = Tables.rowcount(data)
    all_effects = Matrix{Float64}(undef, n_obs, length(vars))
    g_buffer = Vector{Float64}(undef, length(vars))
    
    for i in 1:n_obs
        # Uses FC primitive:
        FormulaCompiler.marginal_effects_eta!(g_buffer, de, coef(model), i; backend)
        all_effects[i, :] = g_buffer
    end
    
    return MarginsResult(mean(all_effects, dims=1), model, data, vars)
end

function margins_se(results::MarginsResult; method=:delta)
    # NEW: Delta method standard errors using FC Jacobians
    # NEW: Bootstrap standard errors using FC scenario system
    # etc.
end
```

### The Real Architecture: Foundation + Interface

**FormulaCompiler.jl** = High-performance mathematical foundation
- All current code stays
- Provides 0-byte, ~50ns computational primitives
- Mathematical operations with no statistical interpretation

**Margins.jl** = Statistical interface and workflows  
- All new code
- Orchestrates FC primitives into statistical workflows
- Adds interpretation, inference, and visualization

### Migration Summary

| Component | Stays in FormulaCompiler.jl | New in Margins.jl |
|-----------|----------------------------|-------------------|
| **All current code** | ✅ Everything stays | ❌ Nothing moves |
| **Derivative computations** | ✅ All mathematical primitives | ❌ No raw math |
| **Scenario system** | ✅ General counterfactual engine | ❌ No override logic |
| **Statistical workflows** | ❌ None exist currently | ✅ All new interfaces |
| **Inference methods** | ❌ None exist currently | ✅ Standard errors, tests, CIs |
| **Visualization** | ❌ None exists currently | ✅ All plotting functions |

The value proposition is **not** about moving existing code, but about **FormulaCompiler.jl providing the zero-allocation computational foundation** that enables Margins.jl to build sophisticated statistical workflows that would be impossible to implement efficiently otherwise.

## Implementation Considerations

### API Compatibility
- FormulaCompiler.jl maintains backward compatibility for all computational primitives
- Margins.jl can iterate rapidly on user interface without affecting foundation
- Clear versioning strategy to manage dependencies

### Performance Preservation
- All current zero-allocation achievements preserved in FormulaCompiler.jl
- Margins.jl adds minimal overhead through careful use of computational primitives
- Benchmarking across both packages to ensure no performance regression

### Documentation Strategy
- FormulaCompiler.jl: Technical documentation focused on implementation
- Margins.jl: User-focused documentation with statistical examples and interpretation
- Cross-references between packages for users needing both perspectives

This architectural separation positions both packages for long-term success in their respective domains while maximizing the utility of the sophisticated computational engine that FormulaCompiler.jl has become.