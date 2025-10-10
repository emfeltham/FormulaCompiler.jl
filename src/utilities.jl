# utilities.jl

"""
    continuous_variables(compiled, data) -> Vector{Symbol}

Identify continuous variables suitable for derivative computation from a compiled formula.

Analyzes compiled operations to distinguish between continuous variables (suitable for
differentiation) and categorical variables (requiring discrete analysis). Essential for
determining valid variable sets for derivative evaluators and marginal effects computation.

# Arguments
- `compiled::UnifiedCompiled`: Compiled formula from `compile_formula(model, data)`  
- `data::NamedTuple`: Data in column-table format (from `Tables.columntable(df)`)

# Returns
- `Vector{Symbol}`: Sorted list of continuous variable names
  - Includes: Float64, Int64, Int32, Int variables used in LoadOp operations
  - Excludes: Variables appearing only in ContrastOp operations (categorical contrasts)
  - Excludes: Boolean variables (treated as categorical regardless of numeric type)

# Classification Algorithm
1. **Operation analysis**: Scan compiled operations for LoadOp vs ContrastOp usage
2. **Type filtering**: Verify variables have Real element types in data
3. **Boolean exclusion**: Remove Bool variables (categorical by convention)
4. **Categorical exclusion**: Remove variables only appearing in contrast operations

# Example
```julia
using FormulaCompiler, GLM, CategoricalArrays

# Mixed variable types
df = DataFrame(
    y = randn(1000),
    price = randn(1000),          # Float64 - continuous
    quantity = rand(1:100, 1000), # Int64 - continuous
    available = rand(Bool, 1000), # Bool - categorical
    category = categorical(rand([\"A\", \"B\", \"C\"], 1000))  # Categorical - categorical
)

model = lm(@formula(y ~ price + quantity + available + category), df)
compiled = compile_formula(model, Tables.columntable(df))

# Identify continuous variables
continuous_vars = continuous_variables(compiled, Tables.columntable(df))
# Returns: [:price, :quantity]

# Use for derivative evaluator construction
de_fd = derivativeevaluator_fd(compiled, Tables.columntable(df), continuous_vars)
de_ad = derivativeevaluator_ad(compiled, Tables.columntable(df), continuous_vars)
```

# Use Cases
- **Pre-validation**: Check variable suitability before building derivative evaluators
- **Automatic selection**: Programmatically identify all differentiable variables
- **Error prevention**: Avoid attempting derivatives on categorical variables
- **Model introspection**: Understand variable roles in compiled formulas

# Implementation Details
- Scans LoadOp operations for direct variable usage (continuous indicators)
- Identifies ContrastOp operations for categorical variable detection
- Applies type checking to ensure Real element types in the actual data
- Returns sorted list for consistent ordering across calls

See also: [`derivativeevaluator_fd`](@ref), [`derivativeevaluator_ad`](@ref), [`derivative_modelrow!`](@ref)
"""
function continuous_variables(compiled::UnifiedCompiled, data::NamedTuple)
    cont = Set{Symbol}()
    cats = Set{Symbol}()
    for op in compiled.ops
        if op isa LoadOp
            Col = typeof(op).parameters[1]
            push!(cont, Col)
        elseif op isa ContrastOp
            Col = typeof(op).parameters[1]
            push!(cats, Col)
        end
    end
    # Remove any categorical columns
    for c in cats
        delete!(cont, c)
    end
    # Keep only columns that exist in data and are Real-typed (but not Bool or Categorical)
    vars = Symbol[]
    for s in cont
        if hasproperty(data, s)
            col = getproperty(data, s)
            # Exclude categorical columns (including CategoricalCounterfactualVector which has UInt eltype)
            is_categorical = col isa Union{CategoricalArray, CategoricalCounterfactualVector}
            if eltype(col) <: Real && !(eltype(col) <: Bool) && !is_categorical
                push!(vars, s)
            end
        end
    end
    sort!(vars)
    return vars
end
