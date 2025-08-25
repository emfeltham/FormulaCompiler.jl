# UnifiedCompiler Formula Decomposition
# Convert StatsModels terms to unified operations

using StatsModels
using MixedModels: RandomEffectsTerm
using StandardizedPredictors: ZScoredTerm

"""
    decompose_formula(formula::FormulaTerm, data_example::NamedTuple) -> (operations, scratch_size, output_size)

**Core Position Mapping Engine**: Converts StatsModels formulas into typed operations with position mappings.

## Position Mapping Algorithm

This function implements the heart of the position mapping system through a multi-stage process:

### Stage 1: Context Initialization
- Creates `CompilationContext` with empty position map and operation list
- Initializes `next_position = 1` for scratch space allocation
- Prepares `output_positions` to track final model matrix columns

### Stage 2: Term Decomposition & Position Allocation  
- Recursively processes each formula term using `decompose_term!`
- **Position Caching**: Reuses positions for identical terms (e.g., `x` appears in `x + x*z`)  
- **Smart Allocation**: 
  - Single terms → single position (`Term(:x)` → position 3)
  - Multi-output terms → consecutive positions (`CategoricalTerm` → positions [4,5,6])
- **Dependency Tracking**: Child terms allocated before parent operations

### Stage 3: Output Position Mapping
- Maps scratch positions to final model matrix columns in order
- Creates `CopyOp{scratch_pos, output_idx}` for each output column
- Ensures output matches `modelmatrix(model)` column ordering

### Stage 4: Operation Ordering
The function maintains **execution order invariants**:
1. **Leaf operations first**: Data loads and constants
2. **Dependency respect**: Operations use only previously computed positions  
3. **Copy operations last**: Transfer from scratch to output

## Position Allocation Examples

```julia
# Formula: y ~ 1 + x + log(z) + x*group
# 
# Term Processing Order & Position Allocation:
# 1. InterceptTerm{true}()     → pos=1  (ConstantOp{1.0, 1})
# 2. Term(:x)                  → pos=2  (LoadOp{:x, 2})  
# 3. Term(:z)                  → pos=3  (LoadOp{:z, 3})
# 4. FunctionTerm(log, [:z])   → pos=4  (UnaryOp{:log, 3, 4})
# 5. CategoricalTerm(:group)   → pos=[5,6] (ContrastOp{:group, (5,6)})
# 6. InteractionTerm([x,group])→ pos=7  (BinaryOp{:*, 2, 5, 7})
#
# Output Position Mapping:
# output[1] = scratch[1]  (CopyOp{1, 1}) - intercept
# output[2] = scratch[2]  (CopyOp{2, 2}) - x  
# output[3] = scratch[4]  (CopyOp{4, 3}) - log(z)
# output[4] = scratch[5]  (CopyOp{5, 4}) - group level 1
# output[5] = scratch[6]  (CopyOp{6, 5}) - group level 2  
# output[6] = scratch[7]  (CopyOp{7, 6}) - x*group
```

## Position Caching & Reuse

The position map (`ctx.position_map`) implements intelligent caching:

```julia
# Formula: y ~ x + x^2 + log(x)
# 
# First encounter: Term(:x) → allocates position 1, caches {:x → 1}
# Second encounter: Term(:x) → returns cached position 1 (no new allocation)  
# Third encounter: Term(:x) → returns cached position 1 (no new allocation)
#
# Result: Only 1 LoadOp{:x, 1} created, reused by all dependent operations
```

## Arguments

- `formula`: StatsModels FormulaTerm (potentially schema-applied)
- `data_example`: Sample data for categorical level inference and validation

## Returns

- `operations::Vector{AbstractOp}`: Ordered list of typed operations
- `scratch_size::Int`: Maximum scratch position used (buffer size needed)  
- `output_size::Int`: Number of model matrix columns (output vector size)

## Invariants Maintained

1. **No position conflicts**: Each scratch position used by exactly one operation
2. **Dependency satisfaction**: Operations only reference previously computed positions  
3. **Output completeness**: Every output position has exactly one copy operation
4. **Cache correctness**: Identical terms always map to identical positions
"""
function decompose_formula(formula::FormulaTerm, data_example::NamedTuple)
    ctx = CompilationContext()
    
    # Process RHS terms (may be wrapped in MatrixTerm after schema application)
    rhs = formula.rhs
    if isa(rhs, MatrixTerm)
        # Model formulas wrap terms in MatrixTerm
        for term in rhs.terms
            positions = decompose_term!(ctx, term, data_example)
            if isa(positions, Int)
                push!(ctx.output_positions, positions)
            elseif isa(positions, Vector{Int})
                append!(ctx.output_positions, positions)
            end
        end
    elseif isa(rhs, AbstractArray)
        # Raw formula with array of terms
        for term in rhs
            positions = decompose_term!(ctx, term, data_example)
            if isa(positions, Int)
                push!(ctx.output_positions, positions)
            elseif isa(positions, Vector{Int})
                append!(ctx.output_positions, positions)
            end
        end
    else
        # Single term on RHS
        positions = decompose_term!(ctx, rhs, data_example)
        if isa(positions, Int)
            push!(ctx.output_positions, positions)
        elseif isa(positions, Vector{Int})
            append!(ctx.output_positions, positions)
        end
    end
    
    # Add copy operations for final output
    for (out_idx, scratch_pos) in enumerate(ctx.output_positions)
        push!(ctx.operations, CopyOp{scratch_pos, out_idx}())
    end
    
    return ctx.operations, ctx.next_position - 1, length(ctx.output_positions)
end

# Check if formula has intercept
function has_intercept(formula::FormulaTerm)
    # Handle single term RHS
    rhs = formula.rhs
    if isa(rhs, AbstractArray)
        for term in rhs
            if term isa InterceptTerm{true}
                return true
            end
        end
    else
        # Single term on RHS
        if rhs isa InterceptTerm{true}
            return true
        end
    end
    return false
end

# Check if formula explicitly excludes intercept
function has_explicit_no_intercept(formula::FormulaTerm)
    rhs = formula.rhs
    if isa(rhs, AbstractArray)
        for term in rhs
            if term isa InterceptTerm{false} || (term isa ConstantTerm && term.n == 0)
                return true
            end
        end
    else
        # Single term on RHS
        if rhs isa InterceptTerm{false} || (rhs isa ConstantTerm && rhs.n == 0)
            return true
        end
    end
    return false
end

# Decompose different term types
function decompose_term!(ctx::CompilationContext, term::Term, data_example)
    # Check cache first
    if haskey(ctx.position_map, term)
        return ctx.position_map[term]
    end
    
    if term.sym == Symbol("1")  # Intercept
        pos = allocate_position!(ctx)
        push!(ctx.operations, ConstantOp{1.0, pos}())
        ctx.position_map[term] = pos
        return pos
    else  # Variable
        pos = allocate_position!(ctx)
        push!(ctx.operations, LoadOp{term.sym, pos}())
        ctx.position_map[term] = pos
        return pos
    end
end

# Handle function terms
function decompose_term!(ctx::CompilationContext, term::FunctionTerm, data_example)
    # Check cache
    if haskey(ctx.position_map, term)
        return ctx.position_map[term]
    end
    
    # Get function name
    func_name = term.f
    if isa(func_name, typeof(exp))
        func_sym = :exp
    elseif isa(func_name, typeof(log))
        func_sym = :log
    elseif isa(func_name, typeof(sqrt))
        func_sym = :sqrt
    elseif isa(func_name, typeof(abs))
        func_sym = :abs
    elseif isa(func_name, typeof(sin))
        func_sym = :sin
    elseif isa(func_name, typeof(cos))
        func_sym = :cos
    else
        # Handle power function specially
        if func_name isa typeof(^)
            func_sym = :^
        else
            error("Unknown function: $func_name")
        end
    end
    
    # Recursively decompose arguments
    arg_positions = Int[]
    for arg in term.args
        pos = decompose_term!(ctx, arg, data_example)
        push!(arg_positions, isa(pos, Int) ? pos : pos[1])  # Handle multi-output
    end
    
    # Create operation based on arity
    out_pos = allocate_position!(ctx)
    
    if length(arg_positions) == 1
        push!(ctx.operations, UnaryOp{func_sym, arg_positions[1], out_pos}())
    elseif length(arg_positions) == 2
        push!(ctx.operations, BinaryOp{func_sym, arg_positions[1], arg_positions[2], out_pos}())
    else
        # For n-ary functions, cascade binary operations
        current = arg_positions[1]
        for i in 2:length(arg_positions)
            new_pos = (i == length(arg_positions)) ? out_pos : allocate_position!(ctx)
            push!(ctx.operations, BinaryOp{func_sym, current, arg_positions[i], new_pos}())
            current = new_pos
        end
    end
    
    ctx.position_map[term] = out_pos
    return out_pos
end

# Handle interaction terms
function decompose_term!(ctx::CompilationContext, term::InteractionTerm, data_example)
    # Check cache
    if haskey(ctx.position_map, term)
        return ctx.position_map[term]
    end
    
    # Get positions for all component terms
    positions = Int[]
    for t in term.terms
        pos = decompose_term!(ctx, t, data_example)
        if isa(pos, Int)
            push!(positions, pos)
        else
            # For multi-output terms (categorical), use first position for now
            # TODO: Handle categorical×continuous properly
            push!(positions, pos[1])
        end
    end
    
    # Generate multiplication operations
    if length(positions) == 2
        out_pos = allocate_position!(ctx)
        push!(ctx.operations, BinaryOp{:*, positions[1], positions[2], out_pos}())
        ctx.position_map[term] = out_pos
        return out_pos
    else
        # Multi-way interaction: cascade multiplications
        current = positions[1]
        for i in 2:length(positions)
            out_pos = allocate_position!(ctx)
            push!(ctx.operations, BinaryOp{:*, current, positions[i], out_pos}())
            current = out_pos
        end
        ctx.position_map[term] = current
        return current
    end
end

# Handle categorical terms
function decompose_term!(ctx::CompilationContext, term::CategoricalTerm, data_example)
    # Check cache
    if haskey(ctx.position_map, term)
        return ctx.position_map[term]
    end
    
    # Get contrast matrix from term
    contrasts = term.contrasts
    contrast_matrix = Matrix{Float64}(contrasts.matrix)  # Ensure it's Float64
    n_levels = size(contrast_matrix, 1)
    n_contrasts = size(contrast_matrix, 2)
    
    # Allocate positions for each contrast
    positions = allocate_positions!(ctx, n_contrasts)
    
    # Create contrast operation with matrix as field value
    push!(ctx.operations, ContrastOp{term.sym, Tuple(positions)}(contrast_matrix))
    
    ctx.position_map[term] = positions
    return positions
end

# Handle constant terms
function decompose_term!(ctx::CompilationContext, term::ConstantTerm, data_example)
    if haskey(ctx.position_map, term)
        return ctx.position_map[term]
    end
    
    pos = allocate_position!(ctx)
    push!(ctx.operations, ConstantOp{Float64(term.n), pos}())
    ctx.position_map[term] = pos
    return pos
end

# Handle InterceptTerm (appears in schema-applied formulas)
function decompose_term!(ctx::CompilationContext, term::InterceptTerm{true}, data_example)
    # Check cache
    if haskey(ctx.position_map, term)
        return ctx.position_map[term]
    end
    
    pos = allocate_position!(ctx)
    push!(ctx.operations, ConstantOp{1.0, pos}())
    ctx.position_map[term] = pos
    return pos
end

# Handle explicit no-intercept
function decompose_term!(ctx::CompilationContext, term::InterceptTerm{false}, data_example)
    # No-intercept term: don't generate any operation
    return Int[]
end

# Handle MatrixTerm (created by lm/glm when building model matrices)
function decompose_term!(ctx::CompilationContext, term::MatrixTerm, data_example)
    # MatrixTerm contains a tuple of terms that form the model matrix
    return decompose_term!(ctx, term.terms, data_example)
end

# Handle tuple of terms (StatsModels expands interactions this way)
function decompose_term!(ctx::CompilationContext, terms::Tuple, data_example)
    positions = Int[]
    for term in terms
        pos = decompose_term!(ctx, term, data_example)
        if isa(pos, Int)
            push!(positions, pos)
        else
            append!(positions, pos)
        end
    end
    return positions
end

# Handle ContinuousTerm (appears in schema-applied formulas)
function decompose_term!(ctx::CompilationContext, term::ContinuousTerm, data_example)
    # Check cache
    if haskey(ctx.position_map, term)
        return ctx.position_map[term]
    end
    
    # ContinuousTerm has a sym field
    pos = allocate_position!(ctx)
    push!(ctx.operations, LoadOp{term.sym, pos}())
    ctx.position_map[term] = pos
    return pos
end

# Handle ZScoredTerm (from StandardizedPredictors)
function decompose_term!(ctx::CompilationContext, term::ZScoredTerm, data_example)
    # For now, just decompose the inner term
    # TODO: Implement proper z-scoring
    return decompose_term!(ctx, term.term, data_example)
end

# Handle RandomEffectsTerm (from MixedModels)
function decompose_term!(ctx::CompilationContext, term::RandomEffectsTerm, data_example)
    # Skip random effects for now - we only compile fixed effects
    # Random effects are handled separately by MixedModels
    return Int[]
end

# Fallback for other term types
function decompose_term!(ctx::CompilationContext, term, data_example)
    # Check if it's a type we can skip
    type_name = string(typeof(term).name.name)
    if type_name == "RandomEffectsTerm"
        # Skip random effects terms from MixedModels
        return Int[]
    end
    
    @warn "Unknown term type: $(typeof(term))"
    # Try to treat as a variable
    pos = allocate_position!(ctx)
    
    # Extract symbol if possible
    if hasproperty(term, :sym)
        push!(ctx.operations, LoadOp{term.sym, pos}())
    else
        error("Cannot decompose term: $term")
    end
    
    ctx.position_map[term] = pos
    return pos
end