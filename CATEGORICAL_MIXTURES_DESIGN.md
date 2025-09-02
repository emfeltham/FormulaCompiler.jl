# Categorical Mixtures Design: Supporting Profile-Based Marginal Effects

## Problem Statement

FormulaCompiler currently supports categorical mixtures (fractional categorical specifications like `mix("A" => 0.3, "B" => 0.7)`) only through the override/scenario system. This creates significant limitations for **profile-based marginal effects** workflows that are essential for statistical packages like Margins.jl.

### Current Limitation

```julia
# What marginal effects packages need:
reference_grid = DataFrame(
    x = [1.0, 2.0, 3.0],
    category = mix("A" => 0.5, "B" => 0.5)  # ❌ Not supported
)
compiled = compile_formula(model, reference_grid)  # ❌ Would fail
modelrow!(output, compiled, reference_grid, 1)     # ❌ Can't evaluate directly

# Current workaround (inefficient):
base_grid = DataFrame(x = [1.0, 2.0, 3.0], category = ["A", "A", "A"])
compiled = compile_formula(model, base_grid)
mixture_obj = MixtureWithLevels(mix("A" => 0.5, "B" => 0.5), ["A", "B"])
scenario = create_scenario("profile", base_grid; category = mixture_obj)
modelrow!(output, compiled, scenario.data, 1)      # Requires override system
```

### Performance Impact

The override-only approach creates several performance bottlenecks:

1. **Compilation overhead**: Separate scenarios needed for each profile specification
2. **Memory overhead**: Multiple scenario objects instead of single compiled formula
3. **Cache inefficiency**: Override system bypasses type specialization benefits
4. **API complexity**: Cannot use simple `modelrow!()` pattern for batch operations

## Design Approaches

### Approach 1: Compile-Time Mixture Support (Recommended)

**Core idea**: Extend the compilation system to recognize and handle mixture specifications at compile time, similar to how categorical contrasts are currently handled.

#### Implementation Strategy

1. **Data preprocessing**: Detect mixture columns during compilation
2. **Type specialization**: Create specialized evaluator types for mixture operations
3. **Execution optimization**: Pre-compute weighted contrast combinations

```julia
# Proposed API
reference_grid = DataFrame(
    x = [1.0, 2.0, 3.0],
    category = mix("A" => 0.5, "B" => 0.5)  # ✅ Supported directly
)
compiled = compile_formula(model, reference_grid)   # ✅ Handles mixtures
modelrow!(output, compiled, reference_grid, 1)     # ✅ Zero-allocation evaluation
```

#### Technical Details

**New compilation path:**
```julia
# In decomposition.jl - detect mixture columns
function detect_mixture_columns(data)
    mixture_cols = Symbol[]
    for (name, col) in pairs(data)
        if is_mixture_column(col)  # Duck typing for MixtureWithLevels
            push!(mixture_cols, name)
        end
    end
    return mixture_cols
end

# In types.jl - new operation type
struct MixtureContrastOp{Col, Positions, Levels, Weights} <: FormulaOperation
    contrast_matrix::Matrix{Float64}
    # Levels and Weights embedded in type for specialization
end

# In execution.jl - optimized mixture evaluation
function execute_operation!(
    scratch::AbstractVector{Float64},
    op::MixtureContrastOp{Col, Positions, Levels, Weights},
    data, row_idx
) where {Col, Positions, Levels, Weights}
    # Pre-computed weighted combination (type-stable)
    for (i, pos) in enumerate(Positions)
        scratch[pos] = 0.0
        for (level_idx, weight) in zip(Levels, Weights)
            scratch[pos] += weight * op.contrast_matrix[level_idx, i]
        end
    end
end
```

#### Advantages
- **Zero-allocation performance**: Maintains FormulaCompiler's core performance characteristics
- **Type stability**: Mixture weights embedded in type parameters
- **Simple API**: Works with standard `modelrow!()` interface
- **Batch efficiency**: Single compilation handles entire reference grid

#### Implementation Scope
- **Medium complexity**: Requires changes to compilation, decomposition, and execution systems
- **Type system integration**: Leverages existing position-mapping architecture
- **Backward compatibility**: Existing categorical handling unchanged

### Approach 2: Runtime Mixture Resolution

**Core idea**: Defer mixture resolution to execution time while maintaining type stability.

```julia
# Mixed compilation - some mixtures resolved at runtime
struct RuntimeMixtureOp{Col, Positions} <: FormulaOperation
    contrast_matrix::Matrix{Float64}
end

function execute_operation!(scratch, op::RuntimeMixtureOp{Col, Positions}, data, row_idx)
    col_value = getproperty(data, Col)[row_idx]
    if is_mixture(col_value)
        # Runtime mixture resolution
        compute_weighted_contrast!(scratch, op.contrast_matrix, col_value, Positions)
    else
        # Standard categorical handling
        standard_categorical_lookup!(scratch, op.contrast_matrix, col_value, Positions)
    end
end
```

#### Advantages
- **Flexibility**: Supports both concrete and mixture specifications in same dataset
- **Incremental implementation**: Can be added alongside existing system

#### Disadvantages
- **Runtime overhead**: Type checking and mixture resolution at execution time
- **Allocation potential**: Harder to guarantee zero-allocation performance
- **Complexity**: Mixed execution paths increase maintenance burden

### Approach 3: Hybrid Preprocessing + Override

**Core idea**: Enhance the scenario system to be more efficient for profile-based workflows.

```julia
# Optimized profile scenario creation
function create_profile_compiled(model, base_data, profile_mixtures::Dict)
    # Pre-create all mixture scenarios
    scenarios = [create_scenario("profile_$i", base_data; profile_mixtures...) 
                 for i in 1:nrows(base_data)]
    
    # Return efficient batch evaluator
    return ProfileCompiledEvaluator(compile_formula(model, base_data), scenarios)
end

struct ProfileCompiledEvaluator
    base_compiled::CompiledFormula
    scenarios::Vector{DataScenario}
end

function modelrow!(output, evaluator::ProfileCompiledEvaluator, row_idx)
    modelrow!(output, evaluator.base_compiled, evaluator.scenarios[row_idx].data, row_idx)
end
```

#### Advantages
- **Minimal core changes**: Builds on existing override system
- **Performance optimization**: Batch scenario creation reduces overhead

#### Disadvantages
- **Memory overhead**: Still requires scenario objects for each profile row
- **API complexity**: Different interface from standard FormulaCompiler patterns
- **Scalability limits**: Memory usage grows with profile grid size

## Recommendation: Approach 1 (Compile-Time Support)

**Rationale:**
1. **Performance alignment**: Maintains FormulaCompiler's zero-allocation guarantees
2. **API consistency**: Works with existing `modelrow!()` patterns
3. **Scalability**: Memory usage independent of profile grid size
4. **Future-proof**: Enables efficient marginal effects implementations

**Implementation Priority:**
1. **Phase 1**: Extend `decomposition.jl` to detect mixture columns
2. **Phase 2**: Create `MixtureContrastOp` operation type
3. **Phase 3**: Implement optimized execution path
4. **Phase 4**: Add comprehensive test suite for mixture workflows

**Integration with Margins.jl:**
```julia
# Efficient marginal effects workflow (post-implementation)
using FormulaCompiler, Margins

# Create reference grid with mixtures
grid = reference_grid(model, data; 
    continuous_vars = [:x => [1, 2, 3]],
    categorical_vars = [:group => mix("A" => 0.5, "B" => 0.5)]
)

# Compile once, evaluate many times
compiled = compile_formula(model, grid)
ames = marginal_effects(compiled, grid, coef(model))  # Zero-allocation batch computation
```

This design would position FormulaCompiler as a robust foundation for advanced statistical computing packages requiring efficient profile-based marginal effects computation.

## Implementation Plan: Compile-Time Mixture Support

- [x] ### Phase 1: Mixture Detection and Data Structures

**Files to modify:**
- `src/compilation/types.jl`
- `src/compilation/decomposition.jl`
- `src/core/utilities.jl`

**Tasks:**
1. **Create mixture detection utilities**
   ```julia
   # In utilities.jl
   is_mixture_column(col) = hasproperty(col[1], :levels) && hasproperty(col[1], :weights)
   extract_mixture_spec(mixture_obj) = (levels=mixture_obj.levels, weights=mixture_obj.weights)
   ```

2. **Define MixtureContrastOp type**
   ```julia
   # In types.jl
   struct MixtureContrastOp{Col, Positions, LevelIndices, Weights} <: FormulaOperation
       contrast_matrix::Matrix{Float64}
   end
   ```

3. **Extend decomposition to detect mixtures**
   ```julia
   # In decomposition.jl
   function detect_mixture_terms(terms, data_example)
       mixture_terms = Term[]
       for term in terms
           if is_categorical_term(term) && has_mixture_data(term, data_example)
               push!(mixture_terms, term)
           end
       end
       return mixture_terms
   end
   ```

**Success criteria:**
- Can detect mixture columns in sample data
- MixtureContrastOp type compiles and is type-stable
- Tests pass for basic mixture detection

- [x] ### Phase 2: Compilation Pipeline Integration

**Files to modify:**
- `src/compilation/decomposition.jl`
- `src/compilation/compilation.jl`

**Tasks:**
1. **Extend term decomposition for mixtures**
   ```julia
   function decompose_categorical_term(term::Term, data_example, position_tracker)
       col_data = get_column_data(term, data_example)
       
       if is_mixture_column(col_data)
           return decompose_mixture_term(term, col_data, position_tracker)
       else
           return decompose_standard_categorical(term, col_data, position_tracker)
       end
   end
   
   function decompose_mixture_term(term, mixture_col, position_tracker)
       # Extract mixture specification from first row (all rows same mixture)
       mixture_spec = extract_mixture_spec(mixture_col[1])
       
       # Get contrast matrix for the categorical levels
       contrast_matrix = build_contrast_matrix(mixture_spec.levels, term)
       
       # Allocate positions for contrast columns
       positions = allocate_positions!(position_tracker, size(contrast_matrix, 2))
       
       # Encode mixture spec in type parameters for specialization
       level_indices = tuple(findfirst.(==(string(l)), mixture_spec.levels) for l in mixture_spec.levels)
       weights = tuple(mixture_spec.weights...)
       
       return MixtureContrastOp{
           get_column_symbol(term),
           tuple(positions...),
           level_indices,
           weights
       }(contrast_matrix)
   end
   ```

2. **Update compilation entry point**
   ```julia
   # In compilation.jl - modify compile_formula to handle mixtures
   function compile_formula(model, data)
       # Validate mixture columns are consistent
       validate_mixture_consistency!(data)
       
       # Continue with existing compilation logic
       # (mixture handling integrated into decomposition)
   end
   ```

**Success criteria:**
- Can compile formulas with mixture columns
- Type parameters correctly encode mixture specifications
- Compilation produces correct position allocations

- [x] ### Phase 3: Execution Engine Implementation

**Files to modify:**
- `src/compilation/execution.jl`

**Tasks:**
1. **Implement mixture execution logic**
   ```julia
   # Specialized execution for compile-time mixtures
   function execute_operation!(
       scratch::AbstractVector{Float64},
       op::MixtureContrastOp{Col, Positions, LevelIndices, Weights},
       data, row_idx
   ) where {Col, Positions, LevelIndices, Weights}
       # Zero out positions
       for pos in Positions
           scratch[pos] = 0.0
       end
       
       # Compute weighted combination (fully unrolled at compile time)
       for (level_idx, weight) in zip(LevelIndices, Weights)
           for (contrast_col, pos) in enumerate(Positions)
               scratch[pos] += weight * op.contrast_matrix[level_idx, contrast_col]
           end
       end
   end
   ```

2. **Optimize for common cases**
   ```julia
   # Specialized methods for binary mixtures, etc.
   function execute_operation!(
       scratch::AbstractVector{Float64},
       op::MixtureContrastOp{Col, Positions, (1, 2), (w1, w2)},
       data, row_idx
   ) where {Col, Positions, w1, w2}
       # Highly optimized binary mixture path
       @inbounds for (i, pos) in enumerate(Positions)
           scratch[pos] = w1 * op.contrast_matrix[1, i] + w2 * op.contrast_matrix[2, i]
       end
   end
   ```

**Success criteria:**
- Zero-allocation execution for mixture operations
- Performance comparable to standard categorical operations
- Correct weighted contrast computation

- [x] ### Phase 4: Data Interface and Validation

**Files to modify:**
- `src/compilation/compilation.jl`
- `src/core/utilities.jl`

**Tasks:**
1. **Add mixture column validation**
   ```julia
   function validate_mixture_consistency!(data)
       for (col_name, col_data) in pairs(data)
           if is_mixture_column(col_data)
               validate_mixture_column!(col_name, col_data)
           end
       end
   end
   
   function validate_mixture_column!(col_name, col_data)
       # Check all rows have same mixture specification
       first_spec = extract_mixture_spec(col_data[1])
       for (i, row_mixture) in enumerate(col_data)
           if i == 1; continue; end
           spec = extract_mixture_spec(row_mixture)
           if spec != first_spec
               error("Inconsistent mixture specification in column $col_name at row $i")
           end
       end
       
       # Validate weights sum to 1.0
       if !isapprox(sum(first_spec.weights), 1.0, atol=1e-10)
           error("Mixture weights in column $col_name do not sum to 1.0: $(first_spec.weights)")
       end
   end
   ```

2. **Helper functions for mixture creation**
   ```julia
   # Convenience functions for creating mixture columns
   function create_mixture_column(mixture_spec, n_rows)
       return fill(mixture_spec, n_rows)
   end
   
   function expand_mixture_grid(base_data, mixture_specs::Dict)
       # Create all combinations of base data with mixture specifications
   end
   ```

**Success criteria:**
- Comprehensive validation catches malformed mixture data
- Clear error messages guide users to correct usage
- Helper functions simplify mixture data creation

- [x] ### Phase 5: Testing and Documentation

**Files to create/modify:**
- `test/test_categorical_mixtures.jl`
- Update existing test files
- Update `CLAUDE.md`

**Tasks:**
1. **Comprehensive test suite**
   ```julia
   @testset "Categorical Mixtures" begin
       # Basic mixture compilation
       @testset "Compilation" begin
           df = DataFrame(
               x = [1.0, 2.0],
               cat = [mix("A" => 0.3, "B" => 0.7), mix("A" => 0.3, "B" => 0.7)]
           )
           model = lm(@formula(y ~ x * cat), training_data)
           compiled = compile_formula(model, Tables.columntable(df))
           @test compiled isa CompiledFormula
       end
       
       # Zero-allocation execution
       @testset "Performance" begin
           output = Vector{Float64}(undef, length(compiled))
           @test @allocated(modelrow!(output, compiled, data, 1)) == 0
       end
       
       # Correctness vs manual weighted combination
       @testset "Correctness" begin
           # Test against manually computed weighted contrast
       end
       
       # Edge cases
       @testset "Edge Cases" begin
           # Single level mixtures, extreme weights, etc.
       end
   end
   ```

2. **Documentation updates**
   - Add mixture examples to `CLAUDE.md`
   - Document performance characteristics
   - Add usage patterns for marginal effects

**Success criteria:**
- 100% test coverage for mixture functionality
- Performance tests verify zero-allocation guarantees
- Documentation enables easy adoption

- [x] ### Phase 6: Integration and Optimization

**Files to modify:**
- Performance optimization across all modified files
- `src/evaluation/derivatives/` (if needed for mixture derivatives)

**Tasks:**
1. **Performance profiling and optimization**
   - Benchmark against standard categorical operations
   - Optimize type inference and specialization
   - Minimize compilation overhead

2. **Integration with existing systems**
   - Ensure derivatives work with mixture operations (if applicable)
   - Test with override system compatibility
   - Validate with complex interaction terms

3. **Memory optimization**
   - Minimize memory footprint of MixtureContrastOp
   - Optimize contrast matrix storage
   - Reduce compilation memory usage

**Success criteria:**
- Mixture operations perform within 10% of standard categorical operations
- Memory usage scales appropriately with mixture complexity
- Full compatibility with existing FormulaCompiler features

### Implementation Timeline

**Week 1-2**: Phase 1 (Mixture Detection)
- Set up basic data structures and detection utilities
- Initial tests for mixture identification

**Week 3-4**: Phase 2 (Compilation Integration)  
- Integrate mixture detection into decomposition pipeline
- Ensure type-stable compilation with mixture specifications

**Week 5-6**: Phase 3 (Execution Engine)
- Implement zero-allocation mixture execution
- Optimize for common mixture patterns

**Week 7**: Phase 4 (Data Interface)
- Add validation and helper functions
- Ensure robust error handling

**Week 8-9**: Phase 5 (Testing)
- Comprehensive test suite development
- Documentation updates

**Week 10**: Phase 6 (Optimization)
- Performance tuning and final integration testing
- Prepare for production use

### Success Metrics

**Performance targets:**
- Mixture operations: ≤110% time of standard categorical operations
- Compilation overhead: ≤20% increase for mixture-containing formulas
- Memory usage: O(1) per mixture specification, not per data row

**Functionality targets:**
- Support arbitrary number of mixture levels
- Handle all contrast types (dummy, effects, helmert, etc.)
- Integrate seamlessly with existing interaction terms
- Maintain zero-allocation guarantees for execution

**Quality targets:**
- 100% test coverage for new functionality
- Comprehensive error handling with clear messages
- Full backward compatibility with existing code