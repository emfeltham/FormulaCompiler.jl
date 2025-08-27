# utilities.jl - Utility functions for derivative operations

"""
    continuous_variables(compiled, data) -> Vector{Symbol}

Return a list of continuous variable symbols present in the compiled ops, excluding
categoricals detected via ContrastOps. Filters by `eltype(data[sym]) <: Real`.
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
    # Keep only columns that exist in data and are Real-typed
    vars = Symbol[]
    for s in cont
        if hasproperty(data, s)
            col = getproperty(data, s)
            if eltype(col) <: Real
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
    return sqrt(gβ' * Σ * gβ)
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
    var::Symbol;
    link=GLM.IdentityLink(),
    backend::Symbol=:fd  # Default to :fd for zero-allocation AME
)
    @assert length(gβ_sum) == length(de)
    
    # Use evaluator's fd_yminus buffer as temporary storage
    gβ_temp = de.fd_yminus
    fill!(gβ_sum, 0.0)
    
    # Accumulate gradients across rows with backend selection
    for row in rows
        if link isa GLM.IdentityLink
            # η case: gβ = J_k (single Jacobian column)
            if backend === :fd
                # Zero-allocation single-column FD (optimal for AME)
                fd_jacobian_column!(gβ_temp, de, row, var)
            elseif backend === :ad
                # Compute full Jacobian then extract column (less efficient but more accurate)
                derivative_modelrow!(de.jacobian_buffer, de, row)
                var_idx = findfirst(==(var), de.vars)
                var_idx === nothing && throw(ArgumentError("Variable $var not found in de.vars"))
                gβ_temp .= view(de.jacobian_buffer, :, var_idx)
            else
                throw(ArgumentError("Invalid backend: $backend. Use :fd or :ad"))
            end
        else
            # μ case: use existing FD-based chain rule function
            # (could extend to AD backend if needed, but FD is zero-allocation)
            me_mu_grad_beta!(gβ_temp, de, β, row, var; link=link)
        end
        gβ_sum .+= gβ_temp
    end
    
    # Average
    gβ_sum ./= length(rows)
    return gβ_sum
end