# UnifiedCompiler Main Compilation
# Entry point for unified formula compilation

include("types.jl")
include("execution.jl")
include("scratch.jl")
include("decomposition.jl")

# Helper to extract fixed effects formula
function get_fixed_effects_formula(model)
    # For MixedModels, extract only fixed effects
    if isdefined(MixedModels, :LinearMixedModel) && isa(model, MixedModels.LinearMixedModel)
        return fixed_effects_form(model)
    elseif isdefined(MixedModels, :GeneralizedLinearMixedModel) && isa(model, MixedModels.GeneralizedLinearMixedModel)
        return fixed_effects_form(model)
    else
        # For GLM and other models, use the full formula
        return StatsModels.formula(model)
    end
end

# The fixed_effects_form function is imported at the module level

"""
    compile_formula(model, data_example::NamedTuple) -> UnifiedCompiled

Primary API for compiling statistical models into high-performance evaluators.

## Position Mapping System

This function implements a **position mapping system** that converts statistical 
formulas into zero-allocation execution plans. The system works in three phases:

### Phase 1: Formula Decomposition
- Extracts the schema-applied formula from the fitted model
- Converts StatsModels terms into typed operations (`LoadOp`, `ConstantOp`, etc.)
- Assigns unique **scratch positions** to intermediate values and **output positions** to final results

### Phase 2: Position Allocation
- Uses `CompilationContext.position_map` to track term → position mappings
- Allocates consecutive scratch positions starting from 1
- Maps each model matrix column to a specific output position

### Phase 3: Type Specialization  
- Embeds all positions as compile-time type parameters
- Creates operations like `LoadOp{:x, 3}()` (load column `:x` into scratch position 3)
- Enables zero-allocation execution through complete type specialization

## Position Mapping Examples

```julia
# Simple formula: y ~ 1 + x
# Position mapping:
# scratch[1] = 1.0          (intercept, ConstantOp{1.0, 1})
# scratch[2] = data.x[row]  (variable x, LoadOp{:x, 2})  
# output[1] = scratch[1]    (CopyOp{1, 1})
# output[2] = scratch[2]    (CopyOp{2, 2})

# Interaction: y ~ x * z  
# Position mapping:
# scratch[1] = data.x[row]     (LoadOp{:x, 1})
# scratch[2] = data.z[row]     (LoadOp{:z, 2}) 
# scratch[3] = scratch[1] * scratch[2]  (BinaryOp{:*, 1, 2, 3})
# output[1] = scratch[1], output[2] = scratch[2], output[3] = scratch[3]

# Function: y ~ log(x)
# Position mapping:
# scratch[1] = data.x[row]     (LoadOp{:x, 1})
# scratch[2] = log(scratch[1]) (UnaryOp{:log, 1, 2})
# output[1] = scratch[2]       (CopyOp{2, 1})
```

## Performance Characteristics

- **Scratch space**: Fixed size allocated once, reused for all rows
- **Type stability**: All positions known at compile time → zero allocations
- **Execution**: Pure array indexing with no dynamic dispatch
- **Memory**: O(max_scratch_positions) + O(output_size) per formula

## Arguments

- `model`: Fitted statistical model (GLM, LMM, etc.) with schema-applied formula  
- `data_example`: NamedTuple with sample data for type inference and schema validation

## Returns

`UnifiedCompiled{T, OpsTuple, ScratchSize, OutputSize}` containing:
- Type-specialized operation tuple
- Pre-allocated scratch buffer  
- Position mappings embedded in operation types
"""
function compile_formula(model, data_example::NamedTuple)
    # Extract schema-applied formula using standard API
    # For MixedModels, this extracts only fixed effects
    formula = get_fixed_effects_formula(model)
    
    # Decompose formula to operations (formula has schema info)
    ops_vec, scratch_size, output_size = decompose_formula(formula, data_example)
    
    # Convert to tuple for type stability
    ops_tuple = Tuple(ops_vec)
    
    # Create specialized compiled formula (Float64 by default)
    return UnifiedCompiled{Float64, typeof(ops_tuple), scratch_size, output_size}(ops_tuple)
end

# Export main functions
export UnifiedCompiled, compile_formula
