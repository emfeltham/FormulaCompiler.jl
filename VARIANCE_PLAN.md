# Variance Implementation Plan for FormulaCompiler.jl

## Executive Summary

Based on the analysis of HOLISTIC_PLAN.md and VARIANCE.md, FormulaCompiler.jl is nearly complete for its role as a computational engine. This document outlines the minimal additions needed to support variance/standard error computations while maintaining the architectural separation between computational primitives (FormulaCompiler.jl) and statistical workflows (future Margins.jl).

## Current State: 95% Complete for Computational Engine Role

### ✅ Already Implemented (Zero-Allocation Primitives)

**Single-column derivatives:**
- `fd_jacobian_column!(Jk, de, row, var; step=:auto)` - zero-allocation single-column FD Jacobian
- `fd_jacobian_column_pos!(Jk, de, row, var_idx)` - positional hot path

**Parameter gradients:**
- `me_eta_grad_beta!(gβ, de, β, row, var)` - η marginal effect gradient: `gβ = J_k`
- `me_mu_grad_beta!(gβ, de, β, row, var; link)` - μ marginal effect gradient with chain rule

**Supporting infrastructure:**
- Link function derivatives: `_dmu_deta(link, η)`, `_d2mu_deta2(link, η)`
- Model integration: `vcov(model)` access verified
- All derivative systems with comprehensive test coverage

## Missing Computational Primitives (For VARIANCE.md Implementation)

### 🔧 Need to Add: 2 Small Functions (~25 lines total)

**1. Delta Method Standard Error Computation**
```julia
"""
    delta_method_se(gβ, Σ)

Compute standard error using delta method: SE = sqrt(gβ' * Σ * gβ)

Arguments:
- `gβ::Vector{Float64}`: Parameter gradient vector  
- `Σ::Matrix{Float64}`: Parameter covariance matrix from model

Returns:
- `Float64`: Standard error

Notes:
- Zero allocations per call
- Implements Var(m) = gβ' Σ gβ where m is marginal effect
"""
function delta_method_se(gβ::AbstractVector{Float64}, Σ::AbstractMatrix{Float64})
    return sqrt(gβ' * Σ * gβ)
end
```

**2. Average Marginal Effects Gradient Accumulator**
```julia
"""
    accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link=IdentityLink(), backend=:fd)

Accumulate parameter gradients across rows for average marginal effects with backend selection.

Arguments:
- `gβ_sum::Vector{Float64}`: Preallocated accumulator (modified in-place)
- `de::DerivativeEvaluator`: Built evaluator
- `β::Vector{Float64}`: Model coefficients
- `rows::AbstractVector{Int}`: Row indices to average over
- `var::Symbol`: Variable for marginal effect
- `link`: GLM link function for μ effects
- `backend::Symbol`: `:fd` (finite differences) or `:ad` (automatic differentiation)

Returns:
- The same `gβ_sum` buffer, containing average gradient: gβ_sum .= (1/n) * Σ_i gβ(i)

Backend Selection:
- `:fd`: Zero allocations, optimal for AME across many rows (default)
- `:ad`: Small allocations, more accurate but less efficient for single-variable gradients
- μ case: Currently uses FD-based chain rule regardless of backend

Notes:
- Zero allocations per call with `:fd` backend after warmup
- Uses temporary buffer from evaluator to avoid allocation
- Supports both η and μ cases based on link function
- For η case with `:ad`: computes full Jacobian then extracts column (less efficient)
"""
function accumulate_ame_gradient!(
    gβ_sum::Vector{Float64},
    de::DerivativeEvaluator,
    β::Vector{Float64},
    rows::AbstractVector{Int},
    var::Symbol;
    link=GLM.IdentityLink(),
    backend::Symbol=:fd  # Default to :fd for zero-allocation AME
)
    @assert length(gβ_sum) == length(de)
    
    # Use evaluator's fd_yminus buffer as temporary storage
    gβ_temp = de.fd_yminus
    fill!(gβ_sum, 0.0)
    
    # Accumulate gradients across rows with backend selection
    for row in rows
        if link isa GLM.IdentityLink
            # η case: gβ = J_k (single Jacobian column)
            if backend === :fd
                # Zero-allocation single-column FD (optimal for AME)
                fd_jacobian_column!(gβ_temp, de, row, var)
            elseif backend === :ad
                # Compute full Jacobian then extract column (less efficient but more accurate)
                derivative_modelrow!(de.jacobian_buffer, de, row)
                var_idx = findfirst(==(var), de.vars)
                var_idx === nothing && throw(ArgumentError("Variable $var not found in de.vars"))
                gβ_temp .= view(de.jacobian_buffer, :, var_idx)
            else
                throw(ArgumentError("Invalid backend: $backend. Use :fd or :ad"))
            end
        else
            # μ case: use existing FD-based chain rule function
            # (could extend to AD backend if needed, but FD is zero-allocation)
            me_mu_grad_beta!(gβ_temp, de, β, row, var; link=link)
        end
        gβ_sum .+= gβ_temp
    end
    
    # Average
    gβ_sum ./= length(rows)
    return gβ_sum
end
```

## Implementation Plan

### Phase 1: Add Variance Primitives to FormulaCompiler.jl (1 day)

**Step 1: Add Functions**
1. Add both functions to `src/evaluation/derivatives/utilities.jl`
2. Export in `src/evaluation/derivatives.jl`
3. Export in `src/FormulaCompiler.jl`

**Step 2: Add Tests**
1. Correctness tests in `test/test_derivatives.jl`:
   - Verify `delta_method_se` against analytical examples
   - Test `accumulate_ame_gradient!` against manual averaging with both `:fd` and `:ad` backends
   - Cross-validate backend consistency between `:fd` and `:ad` for η case
   - Cross-validate with reference implementations
2. Allocation tests in `test/test_derivative_allocations.jl`:
   - Verify `delta_method_se` achieves 0 bytes (always)
   - Verify `accumulate_ame_gradient!` with `:fd` backend achieves 0 bytes after warmup
   - Test `:ad` backend allocation behavior (small allocations expected)
   - Include tight-loop tests for both backends

**Step 3: Documentation**
1. Update VARIANCE.md with complete workflow examples
2. Add docstrings with usage examples
3. Update README.md to reflect complete variance capability

### Phase 2: FormulaCompiler.jl v1.0 Release (1 day)

**API Freeze:**
- All computational primitives finalized
- Comprehensive documentation 
- Performance benchmarks validated
- Full test suite passing (2000+ tests)

**Release Checklist:**
- [ ] Variance primitives implemented and tested
- [ ] All allocation tests passing
- [ ] Documentation updated
- [ ] Performance benchmarks meet targets
- [ ] API compatibility guarantee

## Architectural Alignment with HOLISTIC_PLAN.md

### ✅ FormulaCompiler.jl Role (Computational Engine)

**What Stays (All Current Code):**
- Position-mapped compilation system
- Zero-allocation evaluation primitives  
- Modular derivative system (9 modules)
- Scenario/override system
- Statistical model integrations
- **NEW**: Variance computation primitives

**What FormulaCompiler.jl Will NOT Have:**
- High-level statistical workflows (`margins()`, `margins_se()`)
- User-friendly APIs with defaults and conveniences
- Visualization functions
- Statistical interpretation and reporting
- Hypothesis testing interfaces

### 🔮 Future Margins.jl Role (Statistical Interface)

**What Margins.jl Will Build (All New Code):**
```julia
# High-level workflows using FormulaCompiler primitives
results = margins(model, data; at=:means, vars=:continuous)
se_results = margins_se(results; method=:delta) 
ame = average_marginal_effects(model, data; vars=[:x, :z])

# Statistical inference
test_results = margins_test(results; H0=0, vars=[:x])
ci_results = margins_ci(results; level=0.95)

# Visualization
plot_margins(results; type=:effects, vars=[:x, :z])
margins_table(results; format=:latex, digits=3)
```

These will orchestrate FormulaCompiler.jl primitives into complete statistical workflows with interpretation, visualization, and reporting.

## Performance Preservation

**Zero-Allocation Guarantee Maintained:**
- `delta_method_se`: Pure mathematical computation, 0 bytes
- `accumulate_ame_gradient!`: Reuses evaluator buffers, 0 bytes after warmup
- All existing primitives remain unchanged

**Performance Targets:**
- Core evaluation: ~50ns, 0 bytes
- Single-column FD: ~44ns, 0 bytes  
- Parameter gradients: ~58-79ns, 0 bytes
- **NEW** Delta method SE: ~10ns, 0 bytes
- **NEW** AME accumulation (`:fd` backend): ~(n_rows * 60ns), 0 bytes
- **NEW** AME accumulation (`:ad` backend): ~(n_rows * 500ns), small allocations

## Benefits of This Minimal Approach

### 1. **Maintains Architectural Purity**
- FormulaCompiler.jl remains focused on computational excellence
- No high-level APIs that belong in Margins.jl
- Clear separation enables specialized development streams

### 2. **Enables Complete Variance Workflows**
- Supports all VARIANCE.md use cases with zero-allocation primitives
- Provides foundation for sophisticated Margins.jl statistical interfaces
- Maintains performance while adding functionality

### 3. **Future-Proof Foundation**
- Other packages can build variance methods on same primitives
- Extensible to different statistical domains
- Performance-optimized foundation supports complex workflows

## Success Metrics

**FormulaCompiler.jl v1.0 Success:**
- [ ] All VARIANCE.md computational requirements met
- [ ] Zero-allocation guarantee preserved across all functions
- [ ] <100ns per-row performance maintained
- [ ] 2000+ tests passing with new variance functions
- [ ] API freeze enables stable foundation for other packages

**Future Margins.jl Success (Not This Phase):**
- Complete statistical workflows in <10 lines of code
- Results match established packages (Stata, R margins)
- Intuitive user experience for practitioners

## Backend Compatibility Analysis

### ✅ **`delta_method_se(gβ, Σ)` - Universal Compatibility**
**Works with ANY backend** because it only performs linear algebra on the gradient vector `gβ`, regardless of how `gβ` was computed (AD, FD, analytical, etc.).

### ✅ **`accumulate_ame_gradient!(...)` - Dual Backend Support**

**Backend Selection Strategy:**
- **`:fd` backend (default)**: Zero allocations, optimal for AME across many rows
- **`:ad` backend**: Small allocations, higher accuracy, less efficient for single-variable extractions

**Implementation Details:**
- **η case**: 
  - `:fd`: Uses `fd_jacobian_column!()` - zero allocations, ~44ns per row
  - `:ad`: Computes full Jacobian then extracts column - small allocations, ~500ns per row
- **μ case**: Uses FD-based chain rule (`me_mu_grad_beta!`) regardless of backend
  - Could extend to AD if needed, but FD is already zero-allocation for μ

**Performance Trade-offs:**
- **`:fd`**: Optimal for production AME workflows with many observations
- **`:ad`**: Better for small samples where accuracy is more important than allocation efficiency

**Recommendation**: Default to `:fd` backend since AME typically processes many rows where zero-allocation efficiency matters more than small numerical differences.

## Implementation Priority

**Immediate (This Phase):** Add 2 variance primitives to complete FormulaCompiler.jl computational foundation

**Future (Next Package):** Build Margins.jl statistical interface layer on this foundation

This minimal addition strategy ensures FormulaCompiler.jl achieves its architectural role as the "BLAS of statistical computing" - a high-performance foundation that other packages build upon.