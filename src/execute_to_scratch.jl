# execute_to_scratch.jl

@inline function execute_to_scratch!(
    evaluator::AbstractEvaluator,
    scratch::Vector{Float64},
    scratch_start::Int,
    scratch_end::Int,
    data::NamedTuple,
    row_idx::Int
)
    # Manual type dispatch for type stability - same pattern as CombinedEvaluator
    if evaluator isa ConstantEvaluator
        @inbounds scratch[scratch_start] = evaluator.value
        
    elseif evaluator isa ContinuousEvaluator
        col = evaluator.column
        # APPLY TYPE-STABLE DATA ACCESS (same as main execution)
        val = if col === :x
            data.x[row_idx]
        elseif col === :y
            data.y[row_idx] 
        elseif col === :z
            data.z[row_idx]
        elseif col === :w
            data.w[row_idx]
        elseif col === :t
            data.t[row_idx]
        else
            data[col][row_idx]
        end
        @inbounds scratch[scratch_start] = Float64(val)
        
    elseif evaluator isa CategoricalEvaluator
        # Use pre-extracted level codes - no data[column] access!
        level_codes = evaluator.level_codes
        cm = evaluator.contrast_matrix
        n_levels = evaluator.n_levels
        
        @inbounds begin
            lvl = level_codes[row_idx]
            lvl = lvl < 1 ? 1 : (lvl > n_levels ? n_levels : lvl)
            n = scratch_end - scratch_start + 1
            for j in 1:n
                scratch[scratch_start + j - 1] = cm[lvl, j]
            end
        end
        
    elseif evaluator isa FunctionEvaluator
        func            = evaluator.func
        arg_evaluators  = evaluator.arg_evaluators
        arg_scratch_map = evaluator.arg_scratch_map
        n_args          = length(arg_evaluators)

        vals = ntuple(i -> begin
            arg = arg_evaluators[i]
            if arg isa ConstantEvaluator
                arg.value
            elseif arg isa ContinuousEvaluator
                Float64(data[arg.column][row_idx])
            else
                r = arg_scratch_map[i]
                execute_to_scratch!(arg, scratch,
                                    first(r), last(r),
                                    data, row_idx)
                scratch[first(r)]
            end
        end, n_args)

        @inbounds scratch[scratch_start] = apply_function_safe(func, vals...)
        
    elseif evaluator isa ParametricFunctionEvaluator
        # Keep existing implementation for now
        execute_to_scratch!(evaluator, scratch, scratch_start, scratch_end, data, row_idx)
    elseif evaluator isa InteractionEvaluator
        # evaluate each component first
        for (i, comp) in enumerate(evaluator.components)
            r = evaluator.component_scratch_map[i]
            execute_to_scratch!(comp, scratch, first(r), last(r), data, row_idx)
        end
        # then write the interaction itself into this call’s scratch range
        apply_kronecker_to_scratch_range!(evaluator.kronecker_pattern,
                                        evaluator.component_scratch_map,
                                        scratch,
                                        scratch_start,
                                        scratch_end)
    else
        error("Unknown evaluator type in execute_to_scratch!: $(typeof(evaluator))")
    end
    
    return nothing
end

"""
    apply_kronecker_to_scratch_range!(pattern::Vector{NTuple{N,Int}},
                                     component_scratch_map::Vector{UnitRange{Int}},
                                     scratch::Vector{Float64},
                                     output_start::Int,
                                     output_end::Int) where N

Apply Kronecker pattern directly to a scratch range.
"""
function apply_kronecker_to_scratch_range!(
    pattern::Vector{NTuple{N,Int}},
    maps::Vector{UnitRange{Int}},
    scratch::Vector{Float64},
    sstart::Int,
    send::Int
) where N
    @inbounds for idx in 1:length(pattern)
        prod = 1.0
        inds = pattern[idx]
        # no view: compute each component by direct indexing
        for j in 1:N
            r = maps[j]
            prod *= scratch[first(r) + inds[j] - 1]
        end
        scratch[sstart + idx - 1] = prod
    end
end