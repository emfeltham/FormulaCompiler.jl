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