# utilities.jl - Utility functions for derivative operations

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
de = build_derivative_evaluator(compiled, Tables.columntable(df), continuous_vars)
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

See also: [`build_derivative_evaluator`](@ref), [`derivative_modelrow!`](@ref), [`create_scenario`](@ref)
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
    # Keep only columns that exist in data and are Real-typed (but not Bool)
    vars = Symbol[]
    for s in cont
        if hasproperty(data, s)
            col = getproperty(data, s)
            if eltype(col) <: Real && !(eltype(col) <: Bool)
                push!(vars, s)
            end
        end
    end
    sort!(vars)
    return vars
end

"""
    delta_method_se(gβ, Σ)

Compute standard error using delta method: SE = sqrt(gβ' * Σ * gβ)

Arguments:
- `gβ::AbstractVector{Float64}`: Parameter gradient vector  
- `Σ::AbstractMatrix{Float64}`: Parameter covariance matrix from model

Returns:
- `Float64`: Standard error

Notes:
- Zero allocations per call
- Implements Var(m) = gβ' Σ gβ where m is marginal effect
- Works with gradients computed by any backend (AD, FD, analytical)
"""
function delta_method_se(gβ::AbstractVector{Float64}, Σ::AbstractMatrix{Float64})
    # Zero-allocation computation of sqrt(gβ' * Σ * gβ)
    # Use BLAS dot product to avoid temporary arrays
    n = length(gβ)
    result = 0.0
    @inbounds for i in 1:n
        temp = 0.0
        for j in 1:n
            temp += Σ[i, j] * gβ[j]
        end
        result += gβ[i] * temp
    end
    
    # Debug check for negative variance (should not happen with valid covariance matrix)
    if result < 0.0
        @warn "Negative variance detected in delta method: gβ'Σgβ = $result. " *
              "This suggests numerical issues or invalid covariance matrix. " *
              "Check gradient computation and covariance matrix conditioning."
        return NaN
    end
    
    return sqrt(result)
end

"""
    accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link=IdentityLink(), backend=:fd)

Accumulate parameter gradients across rows for average marginal effects with backend selection.

Arguments:
- `gβ_sum::Vector{Float64}`: Preallocated accumulator (modified in-place)
- `de::DerivativeEvaluator`: Built evaluator
- `β::Vector{Float64}`: Model coefficients
- `rows::AbstractVector{Int}`: Row indices to average over
- `var::Symbol`: Variable for marginal effect
- `link`: GLM link function for μ effects
- `backend::Symbol`: `:fd` (finite differences) or `:ad` (automatic differentiation)

Returns:
- The same `gβ_sum` buffer, containing average gradient: gβ_sum .= (1/n) * Σ_i gβ(i)

Backend Selection:
- `:fd`: Zero allocations, optimal for AME across many rows (default)
- `:ad`: Small allocations, more accurate but less efficient for single-variable gradients
- μ case: Currently uses FD-based chain rule regardless of backend

Notes:
- Zero allocations per call with `:fd` backend after warmup
- Uses temporary buffer from evaluator to avoid allocation
- Supports both η and μ cases based on link function
- For η case with `:ad`: computes full Jacobian then extracts column (less efficient)
"""
function accumulate_ame_gradient!(
    gβ_sum::Vector{Float64},
    de::DerivativeEvaluator,
    β::Vector{Float64},
    rows::AbstractVector{Int},
    var::Symbol,
    link=GLM.IdentityLink(),
    backend::Symbol=:fd  # Default to :fd for zero-allocation AME
)
    @assert length(gβ_sum) == length(de)
    
    # Use evaluator's fd_yminus buffer as temporary storage
    gβ_temp = de.fd_yminus
    fill!(gβ_sum, 0.0)

    # Convert symbol to index once (avoid linear search in hot loop)
    var_idx = findfirst(==(var), de.vars)
    var_idx === nothing && throw(ArgumentError("Variable $var not found in de.vars"))

    # Accumulate gradients across rows with backend selection
    for row in rows
        if link isa GLM.IdentityLink
            # η case: gβ = J_k (single Jacobian column)
            if backend === :fd
                # Zero-allocation single-column FD (optimal for AME)
                fd_jacobian_column!(gβ_temp, de, row, var_idx)
            elseif backend === :ad
                # Compute full Jacobian then extract column (less efficient but more accurate)
                derivative_modelrow!(de.jacobian_buffer, de, row)
                gβ_temp .= view(de.jacobian_buffer, :, var_idx)
            else
                throw(ArgumentError("Invalid backend: $backend. Use :fd or :ad"))
            end
        else
            # μ case: use indexed FD-based chain rule function
            me_mu_grad_beta!(gβ_temp, de, β, row, var_idx, link)
        end
        gβ_sum .+= gβ_temp
    end
    
    # Average
    gβ_sum ./= length(rows)
    return gβ_sum
end