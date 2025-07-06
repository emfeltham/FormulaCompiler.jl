# ============================================================================
# workflows.jl - High-level convenience functions
# ============================================================================

"""
    create_update_plan(formula, data) -> (Matrix, MatrixUpdatePlan)

Create both the initial design matrix and the update plan for efficient
incremental updates.

# Returns
- Design matrix for the current data
- Update plan for future selective updates

# Example
```julia
X, plan = create_update_plan(formula, df)
# ... modify df ...
update_matrix_subset!(X, formula.rhs, df, plan, [:modified_var])
```
"""
function create_update_plan(formula, data)
    rhs = formula.rhs
    tbl = Tables.columntable(data)
    
    # Create the plan
    plan = analyze_dependencies(rhs, data)
    
    # More efficient matrix creation
    n_rows = length(first(values(tbl)))
    X = Matrix{Float64}(undef, n_rows, plan.total_width)
    modelmatrix!(X, rhs, data)
    
    return X, plan
end

"""
    incremental_update!(X, plan, rhs, original_data, modifications) -> X

High-level function for making incremental updates to a design matrix.

# Arguments
- `X`: Design matrix to update
- `plan`: Update plan from `create_update_plan`
- `rhs`: Right-hand side of formula
- `original_data`: Original data (will be copied and modified)
- `modifications`: Dict of variable => new_value pairs

# Example
```julia
X, plan = create_update_plan(formula, df)

# Update x1 to 5.0 and x2 to "high"
incremental_update!(X, plan, formula.rhs, df, Dict(:x1 => 5.0, :x2 => "high"))
```
"""
function incremental_update!(
    X::AbstractMatrix,
    plan::MatrixUpdatePlan,
    rhs,
    original_data,
    modifications::Dict{Symbol, Any}
)
    # Create modified data more efficiently
    modified_data = copy(original_data)
    
    # Process modifications more efficiently
    for (var, val) in modifications
        if val isa AbstractVector
            modified_data[!, var] = val
        else
            # Use fill! for scalar values - more efficient
            modified_data[!, var] .= val
        end
    end
    
    # Update only the affected columns
    changed_vars = collect(Symbol, keys(modifications))  # Type-stable collection
    update_matrix_subset!(X, rhs, modified_data, plan, changed_vars)
    
    return X
end
