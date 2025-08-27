# Derivative Test Plan: Manual Baseline Validation

## Problem Statement

The derivative tests are currently failing non-deterministically with systematic discrepancies between AD (ForwardDiff) and FD (finite differences) methods, particularly for models with interaction terms. Without an independent manual baseline, we cannot determine which implementation is correct.

### Current Issues:
1. **Non-deterministic failures**: Tests use random data without fixed seeds
2. **AD vs FD discrepancies**: Differences too large (0.01-0.05) to be numerical precision
3. **No ground truth**: Tests compare AD and FD against each other, not against known-correct values
4. **Interaction term problems**: Discrepancies appear specifically in `x & group3` interaction terms

## Proposed Solution: Manual Baseline Tests

### Phase 1: Infrastructure Setup

#### 1.1 Create Fixed Test Cases
```julia
# test/test_derivatives_baseline.jl
using Random
Random.seed!(12345)  # Fixed seed for reproducibility

# Create small, interpretable test cases
test_cases = [
    # Simple linear model (no interactions)
    (formula = @formula(y ~ 1 + x + z),
     data = DataFrame(y=[1.0, 2.0], x=[0.5, 1.5], z=[2.0, 3.0]),
     description = "Simple linear model"),
    
    # Model with categorical
    (formula = @formula(y ~ 1 + x + group),
     data = DataFrame(y=[1.0, 2.0, 3.0], x=[0.5, 1.5, 2.5], 
                      group=categorical(["A", "B", "A"])),
     description = "Categorical main effect"),
    
    # Model with interaction
    (formula = @formula(y ~ 1 + x + group + x & group),
     data = DataFrame(y=[1.0, 2.0, 3.0], x=[0.5, 1.5, 2.5],
                      group=categorical(["A", "B", "A"])),
     description = "Interaction term x & group")
]
```

#### 1.2 Manual Derivative Calculator
```julia
function compute_manual_jacobian(compiled, data, row, vars; h=1e-8)
    """
    Compute Jacobian using basic finite differences without any 
    FormulaCompiler derivative infrastructure
    """
    n_terms = length(compiled)
    n_vars = length(vars)
    J = Matrix{Float64}(undef, n_terms, n_vars)
    
    # Base evaluation
    y_base = Vector{Float64}(undef, n_terms)
    compiled(y_base, data, row)
    
    for (j, var) in enumerate(vars)
        # Create perturbed data
        data_plus = deepcopy(data)
        data_minus = deepcopy(data)
        
        # Get current value and type
        val = data[var][row]
        
        # Only perturb if numeric
        if val isa Number
            data_plus[var][row] = val + h
            data_minus[var][row] = val - h
            
            # Evaluate at perturbed points
            y_plus = Vector{Float64}(undef, n_terms)
            y_minus = Vector{Float64}(undef, n_terms)
            compiled(y_plus, data_plus, row)
            compiled(y_minus, data_minus, row)
            
            # Central difference
            J[:, j] = (y_plus .- y_minus) ./ (2h)
        else
            # Non-numeric variables have zero derivative
            J[:, j] .= 0.0
        end
    end
    
    return J
end
```

### Phase 2: Baseline Validation Tests

#### 2.1 Direct Comparison Test
```julia
@testset "Derivative Baseline Validation" begin
    for test_case in test_cases
        @testset "$(test_case.description)" begin
            # Fit model and compile
            model = lm(test_case.formula, test_case.data)
            data = Tables.columntable(test_case.data)
            compiled = compile_formula(model, data)
            vars = continuous_variables(compiled, data)
            
            # Build evaluator for AD and FD
            de = build_derivative_evaluator(compiled, data; vars=vars)
            
            for row in 1:nrow(test_case.data)
                # Manual baseline (ground truth)
                J_manual = compute_manual_jacobian(compiled, data, row, vars)
                
                # AD via ForwardDiff
                J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
                derivative_modelrow!(J_ad, de, row)
                
                # FD via evaluator
                J_fd_eval = Matrix{Float64}(undef, length(compiled), length(vars))
                derivative_modelrow_fd!(J_fd_eval, de, row)
                
                # FD standalone
                J_fd_standalone = Matrix{Float64}(undef, length(compiled), length(vars))
                derivative_modelrow_fd!(J_fd_standalone, compiled, data, row; vars=vars)
                
                # All should match manual baseline
                @test isapprox(J_ad, J_manual; rtol=1e-6, atol=1e-8) 
                    broken=(test_case.description == "Interaction term x & group")
                @test isapprox(J_fd_eval, J_manual; rtol=1e-6, atol=1e-8)
                @test isapprox(J_fd_standalone, J_manual; rtol=1e-6, atol=1e-8)
            end
        end
    end
end
```

#### 2.2 Analytical Test Cases
```julia
@testset "Analytical Derivative Tests" begin
    # Test against known analytical derivatives
    
    # Case 1: y = β₀ + β₁x → ∂y/∂x = β₁
    df = DataFrame(y = [1.0, 2.0], x = [1.0, 2.0])
    model = lm(@formula(y ~ 1 + x), df)
    β = coef(model)
    compiled = compile_formula(model, Tables.columntable(df))
    # Expected: ∂X/∂x = [0, 1] for all rows (intercept doesn't depend on x)
    
    # Case 2: y = β₀ + β₁x + β₂x² → ∂y/∂x = β₁ + 2β₂x
    df = DataFrame(y = [1.0, 4.0, 9.0], x = [1.0, 2.0, 3.0])
    df.x_squared = df.x .^ 2
    model = lm(@formula(y ~ 1 + x + x_squared), df)
    # Expected: ∂X/∂x = [0, 1, 2x[row]] for each row
end
```

### Phase 3: Diagnostic Tools

#### 3.1 Discrepancy Reporter
```julia
function diagnose_derivative_discrepancy(compiled, data, row, vars)
    """
    Compare all derivative methods and report discrepancies
    """
    println("Derivative Diagnostic Report")
    println("="^50)
    
    J_manual = compute_manual_jacobian(compiled, data, row, vars)
    J_ad = # ... compute AD
    J_fd = # ... compute FD
    
    for (j, var) in enumerate(vars)
        println("\nVariable: $var")
        println("  Manual:     ", J_manual[:, j])
        println("  AD:         ", J_ad[:, j])
        println("  FD:         ", J_fd[:, j])
        println("  AD-Manual:  ", J_ad[:, j] .- J_manual[:, j])
        println("  FD-Manual:  ", J_fd[:, j] .- J_manual[:, j])
        println("  Max diff:   ", maximum(abs.(J_ad[:, j] .- J_manual[:, j])))
    end
end
```

#### 3.2 Pattern Analyzer
```julia
function analyze_discrepancy_patterns(test_results)
    """
    Identify patterns in where discrepancies occur
    """
    # Group discrepancies by:
    # - Term type (constant, main effect, interaction)
    # - Variable type (continuous, categorical)
    # - Position in model matrix
    # - Magnitude of discrepancy
end
```

### Phase 4: Root Cause Analysis

Once we have manual baselines, we can:

1. **Identify which method is wrong**: Compare AD and FD against manual baseline
2. **Locate the bug**: Determine if issue is in:
   - AD's handling of dual numbers for interactions
   - FD's override system for categorical variables
   - Position mapping for interaction terms
   - Compilation of interaction operations

3. **Test specific hypotheses**:
   - Does the bug only occur with categorical interactions?
   - Does it depend on the categorical level?
   - Is it related to the position mapping system?
   - Does it occur with all GLM link functions?

### Phase 5: Regression Prevention

#### 5.1 Golden Test Suite
```julia
# test/golden/derivatives_golden.jld2
# Store known-good results for regression detection
golden_results = Dict(
    "simple_linear" => J_manual_simple,
    "categorical" => J_manual_categorical,
    "interaction" => J_manual_interaction
)
```

#### 5.2 Property-Based Tests
```julia
@testset "Derivative Properties" begin
    # Linearity: ∂(af(x))/∂x = a * ∂f(x)/∂x
    # Chain rule: ∂f(g(x))/∂x = f'(g(x)) * g'(x)
    # Symmetry: ∂²f/∂x∂y = ∂²f/∂y∂x (for mixed partials)
end
```

## Implementation Priority

1. **Immediate** (Fix current failures):
   - Add `Random.seed!` to existing tests
   - Create `compute_manual_jacobian` function
   - Add baseline validation for current failing tests

2. **High** (Diagnose root cause):
   - Implement diagnostic reporter
   - Test simple vs interaction terms separately
   - Identify exact location of AD/FD discrepancy

3. **Medium** (Comprehensive validation):
   - Add analytical test cases
   - Create golden test suite
   - Implement pattern analyzer

4. **Low** (Long-term robustness):
   - Property-based tests
   - Performance regression tests
   - Cross-validation with other AD packages

## Success Criteria

- [ ] All derivative methods (AD, FD evaluator, FD standalone) match manual baseline within tolerance
- [ ] Tests are deterministic (same results every run)
- [ ] Root cause of interaction term discrepancy identified
- [ ] Regression test suite prevents future breakage
- [ ] Clear documentation of expected behavior for edge cases

## Notes

- The manual baseline should use only basic Julia operations, no FormulaCompiler infrastructure
- Test with both small (n=3-5) and larger (n=100+) datasets
- Consider numerical stability at extreme values
- Document any legitimate differences between AD and FD (e.g., discontinuities)