# FormulaCompiler.jl Test Suite

This directory contains a comprehensive test suite for FormulaCompiler.jl, organized for easy maintenance and development.

## Test Structure

### Core Test Files

- **`runtests.jl`** - Main test runner that executes all test suites

#### Core Functionality
- **`test_position_mapping.jl`** - Position mapping compilation system
- **`test_models.jl`** - Model correctness and core functionality
- **`test_allocations.jl`** - Performance and zero-allocation validation
- **`test_logic.jl`** - Logic operators (comparisons, boolean negation)
- **`test_tough_formula.jl`** - Complex formula edge cases

#### Counterfactual Analysis System
- **`test_overrides.jl`** - Counterfactual vector functionality and scenario analysis
- **`test_categorical_correctness.jl`** - Detailed categorical counterfactual correctness
- **`test_zero_allocation_overrides.jl`** - Counterfactual vector allocation validation
- **`test_compressed_categoricals.jl`** - Compressed categorical arrays (UInt8, UInt16, UInt32)

#### Categorical Mixtures
- **`test_categorical_mixtures.jl`** - Comprehensive mixture test suite
- **`test_mixture_modelrows.jl`** - Modelrow correctness with mixtures

#### Derivatives System
- **`test_derivatives.jl`** - ForwardDiff and finite differences derivatives
- **`test_links.jl`** - GLM link function derivatives for marginal effects
- **`test_derivative_allocations.jl`** - Derivative performance validation
- **`test_contrast_evaluator.jl`** - Zero-allocation contrast evaluator
- **`test_derivatives_log_profile_regression.jl`** - Regression tests
- **`test_derivatives_domain_edge_cases.jl`** - Edge-case regression tests

#### AD Allocation Validation
- **`test_ad_alloc_formula_variants.jl`** - Formula variant allocation profiling
- **`test_formulacompiler_primitives_allocations.jl`** - Core primitive allocations

#### External Integration
- **`test_standardized_predictors.jl`** - StandardizedPredictors.jl integration
- **`test_glm_integration.jl`** - GLM.jl integration
- **`test_mixedmodels_integration.jl`** - MixedModels.jl comprehensive tests

#### Performance Testing
- **`test_large_dataset_performance.jl`** - Large dataset performance validation

#### Documentation
- **`test_documentation_examples.jl`** - Documentation example validation

### Debugging/Diagnostic Utilities (Not in runtests.jl)

These files are development tools for debugging and profiling, not part of the main test suite:

- **`check_testset_alloc.jl`** - Testset allocation checking utility
- **`check_type_stability.jl`** - Type stability analysis tool
- **`debug_binary_alloc.jl`** - Binary allocation debugging
- **`deep_alloc_trace.jl`** - Deep allocation tracing utility
- **`trace_alloc.jl`** - Allocation tracing tool
- **`verify_hypothesis.jl`** - Hypothesis verification utility

### Helper Files

- **`support/testing_utilities.jl`** - Helper functions for testing
- **`support/generate_large_synthetic_data.jl`** - Large dataset generation
- **`Project.toml`** - Test environment dependencies

## Running Tests

### Using Julia Package Manager

```bash
# Run all tests (recommended)
julia --project=. -e "using Pkg; Pkg.test()"
```

### Run Individual Test Files

```bash
# Core functionality
julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_position_mapping.jl\")"
julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_models.jl\")"
julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_allocations.jl\")"

# Override system
julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_overrides.jl\")"

# Categorical mixtures
julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_categorical_mixtures.jl\")"

# Derivatives
julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_derivatives.jl\")"
julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_links.jl\")"

# Integration
julia --project=. -e "using Pkg; Pkg.instantiate(); using BenchmarkTools; include(\"test/test_standardized_predictors.jl\")"
```

### Interactive Development

```julia
# Start Julia with project
julia --project=.

# Using Revise for interactive development
using Revise
using FormulaCompiler
include("test/test_models.jl")
```

## Test Categories

### 1. Core Compilation System

**`test_position_mapping.jl`** - Tests the unified position-mapping compilation architecture:
- Position-based term compilation and type-specialized evaluators
- Formula decomposition and execution logic
- Categorical variable handling with all contrast types
- Interaction terms (two-way, multi-way, mixed)
- Function transformations (log, exp, sqrt, etc.)
- Complex nested expressions

### 2. Model Correctness

**`test_models.jl`** - Validates correctness against reference implementations:
- GLM model compatibility (LinearModel, GeneralizedLinearModel)
- Formula evaluation accuracy vs `modelmatrix()`
- All categorical contrasts (DummyCoding, EffectsCoding, HelmertCoding, etc.)
- Interaction terms and nested structures
- Edge cases (intercept-only, no-intercept, complex formulas)

**`test_logic.jl`** - Logic operators and comparisons:
- Binary comparisons (`<`, `>`, `<=`, `>=`, `==`, `!=`)
- Boolean negation (`!`) and complex boolean expressions
- Automatic type conversion for logical operations
- Integration with formula compilation

**`test_tough_formula.jl`** - Complex formula edge cases:
- Deeply nested transformations
- Complex multi-way interaction patterns
- Mixed categorical/continuous terms
- Stress testing the compilation system

### 3. Performance and Allocations

**`test_allocations.jl`** - Zero-allocation validation:
- Core evaluation path (0 bytes guaranteed)
- ModelRow interface allocations
- Counterfactual vector system performance
- Comprehensive performance benchmarks across all formula types

**`test_large_dataset_performance.jl`** - Large-scale performance validation:
- 500K+ row datasets with realistic complexity
- Scaling characteristics and memory efficiency
- Performance regression detection
- Real-world usage patterns

### 4. Counterfactual Analysis System

**`test_overrides.jl`** - Counterfactual vector functionality:
- `CounterfactualVector` typed implementation achieving O(1) memory
- Single-row variable substitution without data copying
- Scenario creation with `create_scenario()` and grid generation
- Type-stable variants: `BoolCounterfactualVector`, `FloatCounterfactualVector`, `StringCounterfactualVector`, `CategoricalCounterfactualVector`
- Integration with ModelRow and derivatives interfaces

**`test_categorical_correctness.jl`** - Detailed categorical counterfactual validation:
- Categorical structure preservation (critical for contrast matrices)
- Contrast matrix alignment across all contrast types
- Interaction term correctness with categorical overrides
- Edge cases, error conditions, and validation

**`test_zero_allocation_overrides.jl`** - Counterfactual vector performance validation:
- Zero-allocation guarantees for override evaluation
- Type stability verification across all paths
- O(1) memory efficiency vs O(n) data copying
- Performance benchmarks and regression tests

**`test_compressed_categoricals.jl`** - Compressed categorical array support:
- UInt8, UInt16, UInt32 reference types
- Correct handling in counterfactual scenarios
- Memory-efficient categorical representations
- Integration with contrast coding system


### 5. Categorical Mixtures

**`test_categorical_mixtures.jl`** - Comprehensive mixture system:
- Mixture creation with `mix()` constructor
- Weight validation and normalization
- Multiple mixture variables in single formulas
- Interactions between mixtures and continuous variables
- All contrast types (dummy, effects, helmert, etc.)
- Zero-allocation evaluation performance
- Reference grid creation utilities

**`test_mixture_modelrows.jl`** - ModelRow correctness with mixtures:
- Evaluation accuracy vs manual calculations
- Integration with counterfactual vector system
- Complex formula patterns with mixtures
- Profile-based marginal effects support

### 6. Derivatives System

**`test_derivatives.jl`** - Dual-backend derivatives system:
- **Jacobian computation**: `derivative_modelrow!` / `derivative_modelrow` (AD), `derivative_modelrow_fd!` / `derivative_modelrow_fd` (FD)
- **Marginal effects (η)**: `marginal_effects_eta!` / `marginal_effects_eta` with backend selection (`:ad` or `:fd`)
- **Continuous variable detection**: `continuous_variables()` auto-discovery
- **Cross-backend validation**: AD vs FD numerical agreement (rtol=1e-6, atol=1e-8)
- **Integer variable support**: Automatic Float64 conversion for derivatives
- **Evaluator construction**: `build_derivative_evaluator()` with buffer pre-allocation

**`test_links.jl`** - GLM link function derivatives for marginal effects on μ:
- **API**: `marginal_effects_mu!` / `marginal_effects_mu` with backend selection
- **Link functions**: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt, InverseSquare
- **Chain rule**: Correct application of dμ/dη × ∂η/∂x
- **Numerical validation**: Cross-backend agreement and finite difference verification
- **Zero allocations**: Validated for both AD and FD backends

**`test_derivative_allocations.jl`** - Comprehensive derivative performance validation:
- **Zero-allocation verification**: Core primitive, single-row, multi-row, loop patterns
- **FD backend**: 0 bytes for all formula types (guaranteed)
- **AD backend**: 0 bytes for standard formulas, small allocations (~16 bytes) for boolean predicates
- **Backend recommendations**: Performance and allocation tradeoffs
- **CSV output**: `test/derivative_allocations.csv` for tracking

**`test_contrast_evaluator.jl`** - Discrete contrast evaluation:
- **Contrast API**: `contrast_modelrow!` / `contrast_modelrow` for categorical changes
- **Categorical contrasts**: Compute effects of changing categorical levels
- **Zero-allocation**: Validated in-place evaluation
- **Integration**: Works with counterfactual vector system

**`test_derivatives_log_profile_regression.jl`** - Regression test suite:
- Previously problematic formula patterns
- Log and profile transformations
- Complex nested expressions
- Prevents known issues from reoccurring

**`test_derivatives_domain_edge_cases.jl`** - Domain boundary and stability tests:
- Boundary conditions (zeros, near-zeros, large values)
- Numerical stability across domains
- Graceful error handling
- Link function domain restrictions

### 7. AD Allocation Validation

**`test_ad_alloc_formula_variants.jl`** - Formula variant allocation profiling:
- Systematic testing across different formula patterns
- Allocation source identification
- Performance characteristics by pattern type
- Boolean predicate impact analysis

**`test_formulacompiler_primitives_allocations.jl`** - Core primitive allocation validation:
- Basic FormulaCompiler operations (addition, multiplication, etc.)
- Type stability verification
- Batch scaling behavior
- NamedTuple regression guards


### 8. External Integration

**`test_standardized_predictors.jl`** - StandardizedPredictors.jl integration:
- ZScore transformations in formulas
- Correct handling of standardized predictors
- Integration with FormulaCompiler compilation
- Accuracy validation against manual standardization

**`test_glm_integration.jl`** - GLM.jl integration:
- `LinearModel` compatibility and correctness
- `GeneralizedLinearModel` support (logistic, Poisson, etc.)
- Formula compilation from fitted GLM models
- Coefficient extraction and application
- Link function handling

**`test_mixedmodels_integration.jl`** - MixedModels.jl integration:
- Linear mixed models (fixed effects extraction)
- Random slopes and intercepts (fixed component only)
- Multiple random effects structures
- Complex formula patterns
- Integration with counterfactual analysis

### 9. Documentation

**`test_documentation_examples.jl`** - Documentation validation:
- Example code correctness from docs
- API usage patterns match documentation
- User-facing functionality works as documented
- Prevents documentation drift

## Test Support Utilities

### `support/testing_utilities.jl`

Provides utility functions for testing:
- Allocation testing helpers
- Correctness validation utilities
- Performance benchmarking support
- Test data generation functions

### `support/generate_large_synthetic_data.jl`

Large dataset generation for performance testing:
- 500K+ row synthetic datasets
- Configurable categorical/continuous variables
- Realistic data distributions
- Used by `test_large_dataset_performance.jl`

## Development Workflow

### Adding New Tests

1. **Core compilation**: Add to `test_position_mapping.jl` or `test_models.jl`
2. **Counterfactual analysis**: Add to `test_overrides.jl` or `test_categorical_correctness.jl`
3. **Categorical mixtures**: Add to `test_categorical_mixtures.jl` or `test_mixture_modelrows.jl`
4. **Derivatives**: Add to `test_derivatives.jl`, `test_links.jl`, or create regression test
5. **Performance**: Add to `test_allocations.jl` or `test_derivative_allocations.jl`
6. **Integration**: Add to relevant `test_*_integration.jl` file
7. **Regressions**: Add to appropriate `test_*_regression.jl` or `test_*_edge_cases.jl` file

### Test Organization Principles

- **Isolated tests**: Each test should be independent and not rely on state from other tests
- **Comprehensive coverage**: Test both success paths and failure/error cases
- **Performance validation**: Include allocation and timing checks for hot paths
- **Real-world scenarios**: Use realistic data distributions and formula patterns
- **Edge case coverage**: Test boundary conditions, empty data, extreme values
- **Cross-validation**: Compare against reference implementations when available
- **Regression prevention**: Add tests for previously fixed bugs

### Running Specific Test Sets

```julia
# Start Julia REPL with project
julia --project=.

# Run core tests only
@testset "Core Tests" begin
    include("test/test_position_mapping.jl")
    include("test/test_models.jl")
    include("test/test_allocations.jl")
end

# Run counterfactual system tests
@testset "Counterfactual Tests" begin
    include("test/test_overrides.jl")
    include("test/test_categorical_correctness.jl")
    include("test/test_zero_allocation_overrides.jl")
end

# Run derivative tests
@testset "Derivative Tests" begin
    include("test/test_derivatives.jl")
    include("test/test_links.jl")
    include("test/test_derivative_allocations.jl")
end

# Run categorical mixture tests
@testset "Mixture Tests" begin
    include("test/test_categorical_mixtures.jl")
    include("test/test_mixture_modelrows.jl")
end
```

## Performance Expectations

### Target Performance Metrics (v1.0.0 Achieved)

- **Core evaluation**: ~16ns per row, 0 bytes allocated
- **Derivatives (FD)**: ~65ns Jacobian, 0 bytes allocated
- **Derivatives (AD)**: ~49ns Jacobian, 0 bytes allocated
- **Marginal effects η (FD)**: ~82ns, 0 bytes allocated
- **Marginal effects η (AD)**: ~57ns, 0 bytes allocated
- **Marginal effects μ (FD)**: ~108ns, 0 bytes allocated
- **Marginal effects μ (AD)**: ~83ns, 0 bytes allocated
- **Counterfactual scenarios**: O(1) memory vs O(n) for data copying (>99.999% memory reduction)

### Performance Monitoring

The test suite includes comprehensive performance validation:
- **Zero-allocation verification**: Across all hot paths using `@allocated` checks
- **Timing benchmarks**: For regression detection and performance tracking
- **Memory efficiency**: Counterfactual O(1) vs O(n) validation
- **Backend comparison**: AD vs FD tradeoffs (speed, accuracy, allocations)
- **CSV tracking**: `test/derivative_allocations.csv` records historical performance
- **Large-scale testing**: 500K+ row datasets validate scalability

## Debugging Tips

### Correctness Validation

```julia
# Compare compiled formula against reference implementation
using GLM, DataFrames, Tables
df = DataFrame(y = randn(100), x = randn(100), group = rand(["A", "B"], 100))
model = lm(@formula(y ~ x * group), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Verify single row evaluation
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, 1)
reference = modelmatrix(model)[1, :]
@test output ≈ reference  # Should match exactly

# Verify all rows
for i in 1:length(df.y)
    compiled(output, data, i)
    @test output ≈ modelmatrix(model)[i, :]
end
```

### Allocation Debugging

```julia
using BenchmarkTools

# Check for allocations in core evaluation
@allocated compiled(output, data, 1)  # Should be 0 bytes

# Check derivative allocations
de = build_derivative_evaluator(compiled, data; vars=[:x])
g = Vector{Float64}(undef, 1)
@allocated marginal_effects_eta!(g, de, coef(model), 1; backend=:fd)  # Should be 0

# Detailed profiling with BenchmarkTools
@benchmark compiled($output, $data, 1)
@benchmark marginal_effects_eta!($g, $de, $(coef(model)), 1; backend=:fd)

# Compare backends
@benchmark marginal_effects_eta!($g, $de, $(coef(model)), 1; backend=:ad)
@benchmark marginal_effects_eta!($g, $de, $(coef(model)), 1; backend=:fd)
```

### Common Issues and Solutions

1. **Categorical counterfactual failures**
   - **Problem**: Using `fill()` or array comprehensions breaks categorical structure
   - **Solution**: Always use `copy()` on categorical arrays and modify elements
   ```julia
   # WRONG: breaks categorical encoding
   data_override = merge(data, (group = fill("Treatment", n),))

   # CORRECT: preserves categorical structure
   group_modified = copy(data.group)
   group_modified[row_idx] = "Treatment"
   data_override = merge(data, (group = group_modified,))
   ```

2. **Allocation regressions**
   - **Problem**: Type instabilities introduce allocations
   - **Solution**: Check for `Any` types with `@code_warntype`, ensure concrete types
   - **Tools**: Use `trace_alloc.jl` and `deep_alloc_trace.jl` debugging utilities

3. **Derivative accuracy issues**
   - **Problem**: Numerical derivatives differ from AD
   - **Solution**: Cross-validate backends, check step sizes, verify domain constraints
   - **Validation**: Use `rtol=1e-6, atol=1e-8` for cross-backend agreement

4. **Boolean predicate allocations (AD backend only)**
   - **Problem**: Formulas with `(x > 0)` allocate ~16 bytes per predicate with AD
   - **Solution**: Use `:fd` backend for zero allocations with boolean predicates
   - **Note**: This is expected behavior, not a bug

## Test Statistics

The test suite includes:
- **24 test files** in `runtests.jl` covering all functionality
- **6 debugging/diagnostic utilities** for development use
- **2000+ individual tests** with comprehensive coverage
- **Multiple formula patterns** tested for correctness
- **Zero-allocation validation** across all code paths
- **Integration tests** with Julia ecosystem packages (GLM, MixedModels, StandardizedPredictors, etc.)

### Coverage

- **Complete API coverage** - all public functions tested
- **High line coverage** - most code paths tested
- **Edge case coverage** - boundary conditions validated
- **Performance coverage** - allocation and timing verified
- **Regression protection** - known issues prevented

## Contributing

When adding new features:

1. **Write tests first** - TDD approach preferred
2. **Test edge cases** - not just happy path
3. **Include allocation tests** - ensure zero-allocation guarantees
4. **Validate correctness** - compare against reference implementations
5. **Update documentation** - keep test README current
6. **Run full test suite** - `julia --project=. -e "using Pkg; Pkg.test()"`
