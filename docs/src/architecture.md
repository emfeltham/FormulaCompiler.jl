# Architecture

This page provides a technical overview of FormulaCompiler.jl's architecture and design principles.

## Design Philosophy

FormulaCompiler.jl achieves zero-allocation performance through a fundamental architectural principle: **move expensive computations to compile time, leaving only simple, type-stable operations for runtime**.

### Core Principles

1. **Compile-time specialization**: All complex logic happens during compilation
2. **Type stability**: Every runtime operation has predictable, concrete types
3. **Memory reuse**: Pre-allocate everything possible and reuse across evaluations
4. **Position-based dispatch**: Use compile-time positions instead of runtime symbols

## Two-Phase Compilation System

FormulaCompiler.jl uses a sophisticated two-phase compilation system that balances functionality with performance:

### Phase 1: CompiledFormula (Complete Foundation)
- **Purpose**: Complete formula parsing, analysis, and validation
- **Structure**: Self-contained evaluator trees handling all formula complexity
- **Performance**: Good (~100ns per row)
- **Role**: Primary compilation system that does the hard work

```julia
# Phase 1: Build complete evaluator tree
compiled_complete = compile_formula_complete(model, data)
```

### Phase 2: SpecializedFormula (Performance Optimization)  
- **Purpose**: Maximum performance through type specialization
- **Structure**: Analyzes CompiledFormula and creates tuple-based execution paths
- **Performance**: Exceptional (~50ns per row, zero allocations)
- **Role**: Performance optimization layer built on CompiledFormula

```julia
# Phase 2: Create specialized, zero-allocation version
specialized = compile_formula(compiled_complete)

# Or both phases in one call:
compiled = compile_formula(model, data)
```

**Key Insight**: SpecializedFormula doesn't replace CompiledFormula; it specializes it. This architecture ensures that all functionality is preserved while achieving maximum performance.

## Evaluator System

The core of FormulaCompiler.jl is its evaluator system, which provides self-contained components for different term types:

### Abstract Evaluator Hierarchy

```julia
abstract type AbstractEvaluator end

# Concrete evaluator types
struct ConstantEvaluator <: AbstractEvaluator
struct ContinuousEvaluator <: AbstractEvaluator  
struct CategoricalEvaluator <: AbstractEvaluator
struct FunctionEvaluator <: AbstractEvaluator
struct InteractionEvaluator <: AbstractEvaluator
struct CombinedEvaluator <: AbstractEvaluator
```

Each evaluator contains:
- **Positions**: Where to write output in the result vector
- **Logic**: How to compute its contribution
- **Scratch requirements**: Any temporary memory needed

### Position-Based Dispatch

Instead of runtime symbol lookup:
```julia
# Slow: runtime symbol dispatch
data[:x]  # String/Symbol lookup at runtime
```

FormulaCompiler.jl uses compile-time position dispatch:
```julia
# Fast: compile-time position dispatch  
data[Val{3}()]  # Column 3 known at compile time
```

This pattern enables zero-allocation column access across the entire system.

## Four-Step Specialization Pipeline

The SpecializedFormula system uses a four-step pipeline to achieve maximum performance:

### Step 1: Core Foundation (`step1_specialized_core.jl`)
- **Handles**: Constants and continuous variables  
- **Status**: ✅ Zero allocation achieved
- **Types**: `ConstantData{N}`, `ContinuousData{N, Cols}`
- **Key innovation**: `Val{Column}` compile-time dispatch pattern

### Step 2: Categorical Support (`step2_categorical_support.jl`)
- **Handles**: Categorical variables with contrast matrices
- **Status**: ✅ Zero allocation achieved  
- **Features**: Pre-computed contrast lookups
- **Types**: `SpecializedCategoricalData{N, Positions}`

### Step 3: Functions (`step3_functions.jl`)
- **Handles**: Mathematical functions (log, exp, sqrt, etc.)
- **Status**: 🚧 ~32 bytes allocation (allocation issue)
- **Challenge**: Function arguments need `Val{Column}` pattern propagation
- **Types**: `SpecializedFunctionData{UnaryTuple, IntermediateTuple, FinalTuple}`

### Step 4: Interactions (`step4_interactions.jl`)
- **Handles**: All interaction terms using Kronecker product patterns
- **Status**: 🚧 96-864+ bytes allocation (allocation issue)
- **Challenge**: Complex type propagation while maintaining zero allocations
- **Types**: `CompleteInteractionData{IntermediateTuple, FinalTuple}`

## Type System Architecture

### Tuple-Based Execution

FormulaCompiler.jl uses tuples instead of vectors for core execution to avoid allocations:

```julia
# Instead of: (allocates)
operations = [op1, op2, op3]
for op in operations
    execute(op)  # Runtime dispatch + allocation
end

# Use: (zero allocation)
operations_tuple = (op1, op2, op3)  
execute_recursive(operations_tuple)  # Compile-time recursion with Base.tail()
```

### Hybrid Dispatch Strategy

FormulaCompiler.jl uses an empirically-tuned dispatch strategy:

- **≤25 operations**: Recursive tuple dispatch (fastest for small formulas)
- **>25 operations**: `@generated` function dispatch (better for large formulas)

This hybrid approach handles Julia's compilation heuristics optimally.

## Memory Architecture

### Override System

The scenario system uses `OverrideVector` for memory-efficient "what-if" analysis:

```julia
struct OverrideVector{T} <: AbstractVector{T}
    value::T
    length::Int
end

# 8MB traditional vector vs ~32 bytes OverrideVector
traditional = fill(42.0, 1_000_000)  # 8MB
override = OverrideVector(42.0, 1_000_000)  # ~32 bytes
```

### Scratch Space Management

Complex operations (functions, interactions) require scratch space coordination:

```julia
# Each evaluator declares its scratch requirements
scratch_requirement = (
    intermediate_size = 10,  # Temporary calculations
    final_size = 5           # Output positions
)

# Scratch space is pre-allocated and shared across evaluations
scratch = allocate_scratch(total_requirements)
```

## Column Access Patterns

FormulaCompiler.jl has evolved different column access patterns:

### Pattern 1: `Val{Column}` Dispatch (Zero Allocation ✅)
```julia
# Used in Step 1 - constants and continuous variables
function access_column(data, ::Val{Column}) where Column
    return data[Column]  # Compile-time column selection
end
```

### Pattern 2: Symbol-Based Dispatch (Allocating ❌)
```julia  
# Used in Steps 3-4 - causes allocation issues
function access_column(data, column_symbol::Symbol)
    return data[column_symbol]  # Runtime symbol lookup
end
```

**Current Challenge**: Steps 3 and 4 need to adopt the `Val{Column}` pattern consistently to achieve zero allocation.

## Integration Architecture

### Model Type Support

FormulaCompiler.jl uses a dispatch-based system to handle different model types:

```julia
# GLM.jl models
compile_formula(model::StatsModels.TableRegressionModel, data) = ...

# MixedModels.jl - extract fixed effects only  
compile_formula(model::MixedModel, data) = ...

# Add support for new model types by extending dispatch
```

### Formula Processing Pipeline

1. **Extract formula**: Get the `@formula` from the fitted model
2. **Parse terms**: Convert to evaluator tree using StatsModels.jl
3. **Analyze structure**: Determine required evaluator types
4. **Create evaluators**: Build self-contained evaluation components  
5. **Optimize**: Convert to specialized, zero-allocation form
6. **Package**: Return callable object with optimized evaluation

## Performance Architecture

### Zero-Allocation Execution Path

For zero-allocation evaluation, the execution path is:

1. **Pre-allocated output vector**: `row_vec = Vector{Float64}(undef, n_terms)`
2. **Compile-time dispatch**: All column access uses `Val{Column}` 
3. **Type-stable operations**: Every operation has known concrete types
4. **In-place updates**: All writes go directly to pre-allocated vector
5. **No intermediate allocations**: No temporary arrays or vectors created

### Performance Monitoring

The architecture includes built-in performance monitoring:

```julia
# Allocation detection
@allocated compiled(row_vec, data, 1)  # Should be 0

# Timing analysis
@benchmark $compiled($row_vec, $data, 1)  # Should be ~50ns
```

## Current Challenges

### Technical Debt Areas

1. **Column Access Inconsistency**: Steps 3-4 need `Val{Column}` pattern adoption
2. **Function Integration**: Function arguments bypass zero-allocation system
3. **Type Complexity**: Step 4 has 1,911 lines due to complex type propagation
4. **Scratch Coordination**: Complex scratch space management between steps

### Development Focus

The main development priorities are:

1. **Allocation fixes**: Propagate `Val{Column}` pattern through Steps 3-4
2. **Architecture simplification**: Reduce type system complexity
3. **Test coverage**: Improve coverage for complex formula combinations
4. **Performance optimization**: Further reduce evaluation latency

## Extensibility Points

### Adding New Evaluator Types

```julia
# 1. Define evaluator struct
struct CustomEvaluator <: AbstractEvaluator
    # fields for evaluation logic
end

# 2. Implement evaluation method
function evaluate!(output, evaluator::CustomEvaluator, data, row_idx)
    # custom evaluation logic
end

# 3. Add to compilation pipeline
# (integration with existing system)
```

### Adding New Model Types

```julia
# Extend compilation dispatch for new model types
function compile_formula(model::NewModelType, data)
    # Extract formula from new model type
    formula = get_formula(model)
    
    # Use existing compilation pipeline
    return compile_formula_from_terms(formula.rhs, data)
end
```

## Future Architecture Evolution

### Planned Improvements

1. **Unified column access**: Single `Val{Column}` pattern throughout
2. **Simplified type system**: Reduce complexity while maintaining performance
3. **Extended function support**: More mathematical functions with zero allocation
4. **Parallel evaluation**: Multi-threaded row evaluation for large batches
5. **GPU support**: CUDA-compatible evaluation kernels

### Research Directions

1. **Automatic differentiation**: Zero-allocation automatic derivatives
2. **Sparse formulas**: Optimizations for sparse model matrices
3. **Streaming evaluation**: Constant-memory evaluation of unlimited data
4. **Distributed evaluation**: Evaluation across multiple machines

The architecture of FormulaCompiler.jl represents a careful balance between performance, functionality, and maintainability, with a clear path forward for addressing current limitations while preserving the zero-allocation performance that makes it unique in the Julia statistical ecosystem.