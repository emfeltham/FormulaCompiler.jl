# FormulaCompiler.jl Test Suite

This directory contains a comprehensive test suite for FormulaCompiler.jl, organized for easy maintenance and development.

## Test Structure

### Core Test Files

- **`runtests.jl`** - Main test runner that executes all test suites
- **`test_evaluators.jl`** - Tests for core evaluator functionality
- **`test_compilation.jl`** - Tests for formula compilation and code generation
- **`test_modelrow.jl`** - Tests for modelrow interfaces and performance
- **`test_derivatives.jl`** - Tests for analytical derivative compilation
- **`test_scenarios.jl`** - Tests for data scenario functionality
- **`test_evaluator_trees.jl`** - Tests for evaluator tree analysis
- **`test_integration.jl`** - Integration tests with various Julia packages
- **`test_mixed_models.jl`** - Tests for MixedModels.jl integration
- **`test_performance.jl`** - Performance and allocation tests
- **`test_regression.jl`** - Regression tests for known working cases

### Helper Files

- **`test_helpers.jl`** - Helper functions for testing
- **`benchmark.jl`** - Comprehensive benchmarking script
- **`quick_test.jl`** - Quick smoke test for development
- **`Project.toml`** - Test environment dependencies
- **`Makefile`** - Test automation and convenience commands

## Running Tests

### Quick Start

```bash
# Run all tests
make test

# Run specific test suites
make test-core
make test-derivatives
make test-scenarios
make test-performance
make test-integration

# Quick smoke test
make test-quick
```

### Using Julia directly

```julia
# Run all tests
julia --project=test -e "using Pkg; Pkg.instantiate(); include(\"test/runtests.jl\")"

# Run specific test file
julia --project=test -e "using Test; using FormulaCompiler; include(\"test/test_evaluators.jl\")"
```

### Development workflow

```bash
# Set up development environment
make dev-setup

# Run tests with Revise for interactive development
make dev-test
```

## Test Categories

### 1. Core Functionality Tests

Tests the unified operation pipeline and high-level interfaces:
- Operation execution (LoadOp, ConstantOp, UnaryOp, BinaryOp, ContrastOp, CopyOp)
- Formula compilation and position mapping
- Model row interfaces (`modelrow`, `modelrow!`, `ModelRowEvaluator`)
- Scenario system and overrides

### 2. Compilation Tests (`test_compilation.jl`)

Tests formula compilation and code generation:
- Basic formula compilation
- Categorical variable handling
- Function term compilation
- Interaction term compilation
- Complex formula patterns
- Edge cases (intercept-only, no-intercept)
- Compilation caching

### 3. ModelRow Interface Tests (`test_modelrow.jl`)

Tests all modelrow interfaces:
- Zero-allocation `modelrow!` with pre-compiled formulas
- Convenient `modelrow!` with model objects
- Allocating `modelrow` functions
- `ModelRowEvaluator` object-based interface
- Cache management
- Error handling
- Interface consistency

### 4. Derivative Tests (`test_derivatives.jl`)

Tests analytical derivative compilation:
- Basic derivative evaluators
- Function derivatives (chain rule)
- Binary operation derivatives (product rule, quotient rule)
- Interaction derivatives
- Zero derivative detection
- Numerical validation
- Complex expression derivatives

### 5. Scenario Tests (`test_scenarios.jl`)

Tests data scenario functionality:
- `OverrideVector` implementation
- Categorical override handling
- Scenario creation and mutation
- Scenario grids and collections
- Integration with modelrow interfaces
- Memory efficiency

### 6. Structure and Introspection

Lightweight checks for compiled structure where applicable (column counts, names) and integration behaviors.

### 7. Integration Tests (`test_integration.jl`)

Tests integration with Julia ecosystem:
- GLM.jl (LinearModel, GeneralizedLinearModel)
- StandardizedPredictors.jl (ZScore)
- CategoricalArrays.jl (different categorical types)
- Tables.jl (various table formats)
- StatsModels.jl (formula constructs)
- Function terms and complex formulas

### 8. Mixed Models Tests (`test_mixed_models.jl`)

Tests MixedModels.jl integration:
- Linear mixed models
- Random slopes models
- Multiple random effects
- Interactions in mixed models
- Functions in mixed models
- Generalized linear mixed models
- Fixed effects extraction

### 9. Performance Tests (`test_performance.jl`)

Tests performance characteristics:
- Compilation performance
- Zero allocation evaluation
- Evaluation speed
- Memory usage patterns
- Scaling performance
- Comparative performance vs modelmatrix
- Concurrent performance

### 10. Regression Tests (`test_regression.jl`)

Tests known working cases and prevents regressions:
- Originally problematic cases
- Edge cases from development
- Complex interaction patterns
- Formula parsing edge cases
- Data type variations
- Performance regression detection
- Correctness validation

## Test Helpers

### `test_helpers.jl`

Provides utility functions for testing:
- `create_test_data()` - standardized test data
- `test_model_correctness()` - correctness validation
- `test_zero_allocations()` - allocation testing
- `comprehensive_model_test()` - full model testing
- `run_formula_test_suite()` - batch testing
- `benchmark_against_modelmatrix()` - performance comparison

### Usage Example

```julia
using FormulaCompiler
include("test/test_helpers.jl")

# Create test data
df = create_test_data(1000)

# Run comprehensive test on a model
results = comprehensive_model_test(@formula(y ~ x * group + log(z)), df)

# Run test suite on multiple formulas
formulas = create_standard_test_formulas()
results = run_formula_test_suite(formulas, df, verbose=true)
summarize_test_results(results)
```

## Benchmarking

### Running Benchmarks

```bash
# Run comprehensive benchmarks
make test-benchmark

# Or directly
julia --project=test test/benchmark.jl
```

### Benchmark Categories

1. **Compilation benchmarks** - time to compile formulas
2. **Evaluation benchmarks** - time to evaluate single rows
3. **Comparative benchmarks** - vs modelmatrix performance
4. **Stress tests** - many evaluations
5. **Memory usage** - allocation patterns
6. **Scaling tests** - performance with data size

## Development Workflow

### Adding New Tests

1. **For new evaluators**: Add tests to `test_evaluators.jl`
2. **For new functionality**: Create new test file or add to existing
3. **For regressions**: Add to `test_regression.jl`
4. **For performance**: Add to `test_performance.jl`

### Test Organization Principles

- **Isolated tests** - each test should be independent
- **Comprehensive coverage** - test both success and failure cases
- **Performance validation** - include allocation and timing checks
- **Real-world scenarios** - use realistic data and formulas
- **Edge case coverage** - test boundary conditions

### Running Specific Tests

```julia
# Run just evaluator tests
@testset "Evaluators Only" begin
    include("test_evaluators.jl")
end

# Run with custom data
df = create_test_data(500)
@testset "Custom Data Tests" begin
    # Your tests here
end
```

## Continuous Integration

The test suite is designed to work with CI systems:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    julia --project=test -e "using Pkg; Pkg.instantiate()"
    julia --project=test -e "include(\"test/runtests.jl\")"

- name: Run benchmarks
  run: |
    julia --project=test test/benchmark.jl
```

## Performance Expectations

### Target Performance Metrics

- **Compilation time**: < 100ms for simple formulas, < 1s for complex
- **Evaluation time**: < 100ns for simple, < 1μs for complex
- **Memory allocations**: 0 bytes for evaluation
- **Speedup vs modelmatrix**: 10x+ for single rows

### Performance Monitoring

The test suite includes performance regression detection:
- Compilation time benchmarks
- Evaluation time benchmarks
- Memory allocation verification
- Comparative performance tracking

## Troubleshooting

### Common Issues

1. **Test failures after changes**:
   - Run `make test-regression` to check for regressions
   - Check `test_performance.jl` for performance issues

2. **Memory allocation issues**:
   - Use `@allocated` to debug specific evaluations
   - Check for type instabilities

3. **Compilation failures**:
   - Check evaluator tree construction
   - Verify code generation logic

### Debugging Tips

Prefer simple, observable invariants:
- Compare against `modelmatrix(model)` on sampled rows
- Use `@allocated compiled(row_vec, data, i)` to confirm zero allocations
- Inspect operation counts if needed via internal helpers

## Contributing

When adding new features:

1. **Write tests first** - TDD approach
2. **Test edge cases** - not just happy path
3. **Include performance tests** - ensure no regressions
4. **Update documentation** - keep README current
5. **Run full test suite** - `make test-all`

## Test Statistics

The test suite includes:
- **~15 test files** covering all functionality
- **~200+ individual tests** with comprehensive coverage
- **~30 formula patterns** tested for correctness
- **Performance benchmarks** for regression detection
- **Integration tests** with 6+ Julia packages

### Coverage Goals

- **Complete function coverage** - all public functions tested
- **90%+ line coverage** - most code paths tested
- **Edge case coverage** - boundary conditions tested
- **Performance coverage** - allocation and timing validated
