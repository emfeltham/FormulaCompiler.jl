# ============================================================================
# selective_updates.jl - Efficiently update subsets of design matrix
# ============================================================================

"""
    update_matrix_subset!(X, rhs, data, plan::MatrixUpdatePlan, changed_vars) -> X

Update only the columns of `X` that are affected by changes to `changed_vars`.
This is much more efficient than rebuilding the entire matrix when only a few
variables have changed.

# Arguments
- `X`: Design matrix to update in-place
- `rhs`: Right-hand side of the formula
- `data`: New data (with some variables modified)
- `plan`: Pre-computed dependency information
- `changed_vars`: Vector of variables that have been modified

# Example
```julia
plan = analyze_dependencies(formula.rhs, df)
# ... modify some variables in df ...
update_matrix_subset!(X, formula.rhs, df, plan, [:x1, :x3])
```
"""
function update_matrix_subset!(
    X::AbstractMatrix,
    rhs,
    data,
    plan::MatrixUpdatePlan,
    changed_vars::Vector{Symbol}
)
    tbl = Tables.columntable(data)
    terms = _extract_terms(rhs)
    
    # Track which terms need to be rebuilt
    terms_to_rebuild = Set{Int}()
    
    for var in changed_vars
        if haskey(plan.var_to_terms, var)
            for (term_idx, _) in plan.var_to_terms[var]
                push!(terms_to_rebuild, term_idx)
            end
        end
    end
    
    # Rebuild only the affected terms
    for term_idx in terms_to_rebuild
        term = terms[term_idx]
        
        # Find the column range for this term
        col_start = 1 + sum(plan.term_widths[1:term_idx-1])
        col_end = col_start + plan.term_widths[term_idx] - 1
        col_range = col_start:col_end
        
        # Rebuild this term
        new_cols = StatsModels.modelcols(term, tbl)
        
        # Handle different return types
        if new_cols isa AbstractVector
            X[:, col_range] .= reshape(new_cols, :, 1)
        else
            X[:, col_range] .= new_cols
        end
    end
    
    return X
end
