# Statistical Functions Analysis for FormulaCompiler

## Current Capabilities

FormulaCompiler has extensive support for arbitrary formula complexity, including complex interactions, nested functions, and multi-way interactions. The test suite includes formulas like:
- `x * y * group3 + log(z) * group4` - mixed interactions with functions
- `exp(x) * y * group3 * group4` - four-way interactions with functions  
- `log(abs(z)) + group3` - nested function calls
- `exp(x) * log(z)` - function-to-function interactions

## Currently Supported Functions

FormulaCompiler currently supports the following mathematical functions in formula expressions:

### Mathematical Functions
- `exp` - exponential function (e^x)
- `log` - natural logarithm
- `log1p` - log(1+x), numerically stable for small x
- `sqrt` - square root
- `abs` - absolute value

### Trigonometric Functions
- `sin` - sine
- `cos` - cosine

### Arithmetic Operations
- `+` - addition
- `-` - subtraction (binary and unary negation)
- `*` - multiplication
- `/` - division
- `^` - exponentiation

## Missing Functions Analysis

Based on analysis of common statistical modeling practices and R/Julia formula usage, the following functions are commonly used but not yet supported:

### High Priority Functions

These functions appear frequently in statistical models and scientific applications:

#### Logarithmic Functions
- `log10` - base-10 logarithm
  - **Use case**: Scientific data, pH scales, decibel measurements
  - **Example**: `y ~ log10(concentration)`
  - **Implementation**: `log10(x)` in Julia

- `log2` - base-2 logarithm  
  - **Use case**: Genomics, information theory, binary data
  - **Example**: `expression ~ log2(fold_change)`
  - **Implementation**: `log2(x)` in Julia

- `expm1` - exp(x) - 1
  - **Use case**: Numerical stability counterpart to log1p
  - **Example**: `rate ~ expm1(small_coefficient)`
  - **Implementation**: `expm1(x)` in Julia

#### Trigonometric Functions
- `tan` - tangent
  - **Use case**: Completes basic trigonometric set, periodic modeling
  - **Example**: `y ~ tan(phase_angle)`
  - **Implementation**: `tan(x)` in Julia

### Medium Priority Functions

These functions are occasionally useful in statistical modeling:

#### Hyperbolic Functions
- `tanh` - hyperbolic tangent
  - **Use case**: bounded transformations, some link functions
  - **Example**: `response ~ tanh(linear_predictor)`
  - **Implementation**: `tanh(x)` in Julia

- `sinh` - hyperbolic sine
  - **Use case**: Some specialized transformations
  - **Implementation**: `sinh(x)` in Julia

- `cosh` - hyperbolic cosine
  - **Use case**: Some specialized transformations  
  - **Implementation**: `cosh(x)` in Julia

#### Utility Functions
- `sign` - sign function
  - **Use case**: Creating indicator variables, piecewise functions
  - **Example**: `y ~ x * sign(threshold - z)`
  - **Implementation**: `sign(x)` returns -1, 0, or 1

#### Rounding Functions
- `round` - round to nearest integer
  - **Use case**: Discretization, binning continuous variables
  - **Example**: `category ~ round(continuous_score)`
  - **Implementation**: `round(x)` in Julia

- `floor` - round down to integer
  - **Use case**: Discretization, age groups
  - **Example**: `age_group ~ floor(age / 10)`
  - **Implementation**: `floor(x)` in Julia

- `ceil` - round up to integer
  - **Use case**: Discretization, binning
  - **Example**: `bins ~ ceil(value / bin_size)`
  - **Implementation**: `ceil(x)` in Julia

### Low Priority Functions

These functions are rarely used directly in statistical formulas:

#### Inverse Trigonometric Functions
- `asin`, `acos`, `atan` - inverse trigonometric functions
- **Use case**: Specialized transformations, rare in typical statistical models

#### Additional Logarithmic Bases
- `logb(x, base)` - logarithm with arbitrary base
- **Use case**: Very specialized applications
- **Note**: Can be computed as `log(x) / log(base)`

## Implementation Strategy

### Adding New Functions

To add a new function to FormulaCompiler, two files need to be updated:

1. **Function Recognition** (`src/compilation/decomposition.jl`):
   ```julia
   elseif isa(func_name, typeof(new_function))
       func_sym = :new_function
   ```

2. **Function Execution** (`src/compilation/execution.jl`):
   ```julia
   @inline function execute_op(::UnaryOp{:new_function, In, Out}, scratch, data, row_idx) where {In, Out}
       scratch[Out] = new_function(scratch[In])
   end
   ```

3. **Documentation Update** (`src/compilation/types.jl`):
   - Add to the supported functions list

### Validation Strategy

For each new function, validation should include:
- Basic functionality test with simple formula
- Integration with complex formulas (interactions, transformations)
- Derivative support (if applicable)
- Performance validation (zero allocations)

### Example Implementation Template

```julia
# In decomposition.jl
elseif isa(func_name, typeof(log10))
    func_sym = :log10

# In execution.jl  
@inline function execute_op(::UnaryOp{:log10, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = log10(scratch[In])
end

# Test
julia> using FormulaCompiler, GLM, DataFrames, Tables
julia> df = DataFrame(y = randn(10), x = rand(10) .+ 1)
julia> model = lm(@formula(y ~ log10(x)), df)
julia> compiled = compile_formula(model, Tables.columntable(df))
julia> # Should work without errors
```

## Recommendations

### Immediate Implementation
The following functions should be implemented first due to their high usage in statistical applications:
1. `log10` - extremely common in scientific contexts
2. `log2` - important for genomics and information theory
3. `expm1` - numerical stability improvement
4. `tan` - completes basic trigonometric functions

### Future Consideration
- `tanh`
- `sign` for indicator creation
- Basic rounding functions for discretization

### Not Recommended
- Inverse trigonometric functions (rarely used in statistical formulas)
- Complex mathematical functions not commonly appearing in GLM contexts

## Notes

- All functions should maintain FormulaCompiler's zero-allocation performance guarantees
- Functions should be compatible with the existing derivative system
- Priority should be given to functions commonly found in R's formula interface and statistical modeling packages
- Consider numerical stability implications (like log1p/expm1 pair)
