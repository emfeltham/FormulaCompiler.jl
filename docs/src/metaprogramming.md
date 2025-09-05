# Metaprogramming in FormulaCompiler.jl

## Overview

FormulaCompiler.jl uses targeted metaprogramming to achieve zero-allocation evaluation for statistical formulas of arbitrary complexity. This document explains when, why, and how metaprogramming is employed to bypass Julia's inherent limitations while maintaining type stability and performance.

## Design Philosophy

### Metaprogramming as a Precision Tool

The package follows a **"metaprogramming as last resort"** philosophy:

- **Prefer natural Julia**: Use recursion, tuples, and type parameters when possible
- **Metaprogramming only when necessary**: Apply when Julia's built-in mechanisms hit limits
- **Targeted solutions**: Use the minimal metaprogramming needed to solve specific problems
- **Maintain simplicity**: Avoid complex code generation that's hard to understand or maintain

### Core Principle: Compile-Time Specialization

All metaprogramming serves a single goal: **embed runtime decisions into compile-time type specialization** to eliminate allocations and dynamic dispatch.

## Metaprogramming Use Cases

### 1. Large Formula Execution

**Problem**: Julia's tuple specialization is heuristic-based and typically fails beyond 25-35 operations, causing performance degradation for complex statistical formulas.

**Location**: `src/compilation/execution.jl`

**Solution**: Hybrid dispatch strategy with @generated fallback.

```julia
@inline function execute_ops(ops::Tuple, scratch, data, row_idx)
    if length(ops) <= RECURSION_LIMIT  # 25 operations
        # Natural Julia recursion (preferred)
        execute_ops_recursive(ops, scratch, data, row_idx)
    else
        # Metaprogramming fallback (forced specialization)
        execute_ops_generated(ops, scratch, data, row_idx)
    end
end
```

The `@generated` function forces complete unrolling:

```julia
@generated function execute_ops_generated(
    ops::Tuple{Vararg{Any,N}}, 
    scratch::AbstractVector{T}, 
    data::NamedTuple, 
    row_idx::Int
) where {N, T}
    # Build expressions for each operation at compile time
    exprs = Expr[]
    for i in 1:N
        push!(exprs, :(execute_op(ops[$i], scratch, data, row_idx)))
    end
    
    return quote
        $(exprs...)
        nothing
    end
end
```

**Result**: Zero allocations for formulas with 100+ terms, identical performance to small formulas.

### 2. Zero-Allocation Finite Differences

**Problem**: Computing derivatives via finite differences requires loops over variables, creating allocation pressure and dispatch overhead.

**Location**: `src/evaluation/derivatives/finite_diff.jl`

**Solution**: Complete loop unrolling at compile time using type-level variable count.

```julia
@generated function _derivative_modelrow_fd_auto!(
    J::AbstractMatrix{Float64},
    de::DerivativeEvaluator{T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC},
    row::Int,
) where {T, Ops, S, O, NTBase, NTMerged, NV, ColsT, G, JC, GS, GC}
    N = NV  # Extract number of variables from type parameter
    stmts = Expr[]
    
    # Initialize buffers
    push!(stmts, :(yplus = de.fd_yplus))
    push!(stmts, :(yminus = de.fd_yminus)) 
    push!(stmts, :(xbase = de.fd_xbase))
    push!(stmts, :(nterms = length(de)))
    
    # Unroll variable extraction loop
    for j in 1:N
        push!(stmts, :(@inbounds xbase[$j] = de.fd_columns[$j][row]))
    end
    
    # Unroll override setup loop
    for i in 1:N
        push!(stmts, :(@inbounds de.overrides[$i].row = row))
    end
    
    # Unroll main finite difference computation
    for j in 1:N
        push!(stmts, :(x = xbase[$j]))
        
        # Reset all overrides to base values
        for k in 1:N
            push!(stmts, :(@inbounds de.overrides[$k].replacement = xbase[$k]))
        end
        
        # Compute step size
        push!(stmts, :(h = cbrt(eps(Float64)) * max(abs(x), 1.0)))
        
        # Forward perturbation
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x + h))
        push!(stmts, :(de.compiled_dual(yplus, de.data_over_dual, row)))
        
        # Backward perturbation  
        push!(stmts, :(@inbounds de.overrides[$j].replacement = x - h))
        push!(stmts, :(de.compiled_dual(yminus, de.data_over_dual, row)))
        
        # Central difference computation
        push!(stmts, :(inv_2h = 1.0 / (2.0 * h)))
        push!(stmts, quote
            @fastmath for i in 1:nterms
                @inbounds J[i, $j] = (yplus[i] - yminus[i]) * inv_2h
            end
        end)
    end
    
    return quote
        $(stmts...)
        nothing
    end
end
```

**Key Benefits**:
- **Zero allocations**: No dynamic arrays or temporary storage
- **No dispatch overhead**: All variable access patterns embedded at compile time  
- **Optimal step sizing**: Mathematical step size computed once per variable
- **Type stability**: All array accesses use compile-time indices

### 3. Output Buffer Management

**Problem**: Copying results from scratch buffers to output vectors can allocate if done generically.

**Location**: `src/compilation/execution.jl`

**Solution**: Generate copy operations with fixed indices.

```julia
@generated function copy_outputs_generated!(
    ops::Tuple{Vararg{Any,N}}, 
    output::AbstractVector{T}, 
    scratch::AbstractVector{T}
) where {N, T}
    exprs = Expr[]
    for i in 1:N
        # Extract output position from operation type
        if hasfield(typeof(ops.parameters[i]), :output_pos)
            pos = ops.parameters[i].output_pos
            push!(exprs, :(@inbounds output[$pos] = scratch[$pos]))
        end
    end
    
    return quote
        $(exprs...)
        nothing
    end
end
```

## Performance Impact

The figures below are illustrative and hardware-dependent. See the Benchmark Protocol for environment setup and reproduction guidance.

### Metaprogramming Effectiveness

The metaprogramming eliminates allocations and dynamic dispatch in hot paths and preserves small-formula performance for large formulas. See the measured results on the index page and the Benchmark Protocol for how to reproduce them on your hardware.

### Compilation Time Trade-offs

Metaprogramming increases first-compilation latency and compiled code size modestly for complex formulas; subsequent runs use cached code without additional cost. For statistical applications where a formula is compiled once and evaluated many times, this trade‑off is favorable.

## Implementation Patterns

### Pattern 1: Type-Driven Generation

Extract compile-time constants from type parameters:

```julia
@generated function my_function(data::MyType{N, Positions}) where {N, Positions}
    # N and Positions are compile-time constants
    # Generate code using these values
end
```

### Pattern 2: Tuple Length Unrolling

Generate code for each tuple element:

```julia
@generated function process_tuple(ops::Tuple{Vararg{Any,N}}) where N
    exprs = Expr[]
    for i in 1:N
        push!(exprs, :(process_element(ops[$i])))
    end
    return quote; $(exprs...); end
end
```

### Pattern 3: Nested Loop Flattening

Convert nested runtime loops into unrolled compile-time sequences:

```julia
@generated function nested_computation(data::MyType{NVars, NTerms}) where {NVars, NTerms}
    stmts = Expr[]
    for var in 1:NVars
        for term in 1:NTerms
            push!(stmts, :(computation($var, $term)))
        end
    end
    return quote; $(stmts...); end
end
```

## Best Practices

### When to Use Metaprogramming

**Use metaprogramming when**:
- Julia's built-in mechanisms hit limits (tuple specialization, inference)
- Runtime dispatch causes measurable allocation or performance issues
- Loop bounds are known at compile time and unrolling provides benefits
- Type parameters carry sufficient information for code generation

**Avoid metaprogramming when**:
- Natural Julia code achieves the same performance
- Code generation complexity outweighs benefits
- Debugging or maintenance becomes significantly harder
- Compilation time becomes prohibitive

### Code Generation Guidelines

**Structure generated code clearly**:
```julia
@generated function my_function(args...)
    # 1. Extract compile-time information
    N = get_compile_time_constant(args...)
    
    # 2. Build expressions systematically
    setup_exprs = [...]
    loop_exprs = [generate_loop(i) for i in 1:N]
    cleanup_exprs = [...]
    
    # 3. Return well-structured quote block
    return quote
        $(setup_exprs...)
        $(loop_exprs...)
        $(cleanup_exprs...)
        nothing  # Always explicit return
    end
end
```

**Validate generated code**:
```julia
# Include debug utilities
@generated function my_function(args...)
    code = generate_my_code(args...)
    
    # Optional: pretty-print generated code during development
    @static if DEBUG_METAPROGRAMMING
        @info "Generated code:" code
    end
    
    return code
end
```

### Fallback Strategies

Always provide non-metaprogramming fallbacks:

```julia
function my_api_function(args...)
    if should_use_metaprogramming(args...)
        generated_version(args...)
    else
        fallback_version(args...)
    end
end
```

## Integration with Broader Architecture

### Position Mapping Preservation

All metaprogramming maintains the package's core position mapping invariants:

- **Compile-time positions**: All array indices embedded as constants
- **Type stability**: Generated code preserves input/output types
- **Zero allocation**: No dynamic memory management in generated paths

### Testing Generated Code

Generated functions require special testing considerations:

```julia
@testset "Generated Functions" begin
    # Test various type parameter combinations
    for N in [1, 5, 10, 50, 100]
        data = create_test_data(N)
        
        # Test correctness
        @test generated_result(data) ≈ reference_result(data)
        
        # Test allocations
        @test @allocated(generated_version(data)) == 0
        
        # Test performance
        @test @elapsed(generated_version(data)) < performance_threshold
    end
end
```

## Future Considerations

### Evolution Strategy

The metaprogramming approach is designed to be **incrementally replaceable**:

- **Compiler improvements**: If Julia's tuple specialization improves, generated functions can be simplified
- **New language features**: Future Julia versions may provide better alternatives
- **Performance monitoring**: Continuous benchmarking ensures metaprogramming remains beneficial

### Maintenance Approach

- **Isolated complexity**: All metaprogramming confined to specific, well-documented functions
- **Clear interfaces**: Generated functions provide the same API as non-generated alternatives
- **Comprehensive testing**: Extra validation for generated code paths

## Conclusion

FormulaCompiler.jl's metaprogramming serves a specific, measurable purpose: achieving zero-allocation evaluation for arbitrarily complex statistical formulas. The approach is conservative, targeted, and provides clear performance benefits while maintaining code clarity and maintainability.

The key insight is that **metaprogramming enables compile-time specialization** that would be impossible through Julia's standard mechanisms alone, unlocking performance critical for statistical computing applications where the same formula is evaluated thousands or millions of times.
