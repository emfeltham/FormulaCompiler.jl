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
        @inbounds scratch[scratch_start] = Float64(data[col][row_idx])
        
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
        # Cache field accesses
        arg_evaluators = evaluator.arg_evaluators
        arg_scratch_map = evaluator.arg_scratch_map
        func = evaluator.func
        
        n_args = length(arg_evaluators)

        if n_args == 1
            c1 = arg_evaluators[1]
            v1 = c1 isa ConstantEvaluator ? c1.value :
                 (c1 isa ContinuousEvaluator ? Float64(data[c1.column][row_idx]) :
                   begin
                     r1 = arg_scratch_map[1]
                     execute_to_scratch!(c1, scratch, first(r1), last(r1), data, row_idx)
                     scratch[first(r1)]
                   end)
            result = apply_function_safe(func, v1)

        elseif n_args == 2
            c1, c2 = arg_evaluators
            # first arg
            v1 = c1 isa ConstantEvaluator ? c1.value :
                 (c1 isa ContinuousEvaluator ? Float64(data[c1.column][row_idx]) :
                   begin
                     r1 = arg_scratch_map[1]
                     execute_to_scratch!(c1, scratch, first(r1), last(r1), data, row_idx)
                     scratch[first(r1)]
                   end)
            # second arg
            v2 = c2 isa ConstantEvaluator ? c2.value :
                 (c2 isa ContinuousEvaluator ? Float64(data[c2.column][row_idx]) :
                   begin
                     r2 = arg_scratch_map[2]
                     execute_to_scratch!(c2, scratch, first(r2), last(r2), data, row_idx)
                     scratch[first(r2)]
                   end)
            result = apply_function_safe(func, v1, v2)

        elseif n_args == 3
            c1, c2, c3 = arg_evaluators

            # v1
            v1 = if c1 isa ConstantEvaluator
                c1.value
            elseif c1 isa ContinuousEvaluator
                Float64(data[c1.column][row_idx])
            else
                r1 = arg_scratch_map[1]
                execute_to_scratch!(c1, scratch, first(r1), last(r1), data, row_idx)
                scratch[first(r1)]
            end

            # v2
            v2 = if c2 isa ConstantEvaluator
                c2.value
            elseif c2 isa ContinuousEvaluator
                Float64(data[c2.column][row_idx])
            else
                r2 = arg_scratch_map[2]
                execute_to_scratch!(c2, scratch, first(r2), last(r2), data, row_idx)
                scratch[first(r2)]
            end

            # v3
            v3 = if c3 isa ConstantEvaluator
                c3.value
            elseif c3 isa ContinuousEvaluator
                Float64(data[c3.column][row_idx])
            else
                r3 = arg_scratch_map[3]
                execute_to_scratch!(c3, scratch, first(r3), last(r3), data, row_idx)
                scratch[first(r3)]
            end

            result = apply_function_safe(func, v1, v2, v3)

        else
            error("FunctionEvaluator with $n_args args – should be handled by ParametricFunctionEvaluator")
        end

        @inbounds scratch[scratch_start] = result
        
    elseif evaluator isa ParametricFunctionEvaluator
        # Keep existing implementation for now
        execute_to_scratch!(evaluator, scratch, scratch_start, scratch_end, data, row_idx)
        
    else
        error("Unknown evaluator type in execute_to_scratch!: $(typeof(evaluator))")
    end
    
    return nothing
end

#### OLD ####

#=
@inline function execute_to_scratch!(
    eval::ParametricFunctionEvaluator{F,N},
    scratch::Vector{Float64},
    scratch_start::Int,
    scratch_end::Int,
    data::NamedTuple,
    row_idx::Int
) where {F,N}
    # Build an NTuple of argument values without heap allocations
    vals = ntuple(i -> begin
        child = eval.arg_evaluators[i]
        if is_direct_evaluatable(child)
            # constant or raw column
            child isa ConstantEvaluator ? child.value : Float64(data[child.column][row_idx])
        else
            # nested evaluator: recurse to its scratch slot
            r = eval.arg_scratch_map[i]
            execute_to_scratch!(child, scratch, first(r), last(r), data, row_idx)
            scratch[first(r)]
        end
    end, Val{N}())

    # Call the user function with a statically known tuple (no runtime alloc)
    result = apply_function_safe(eval.func, vals...)
    @inbounds scratch[scratch_start] = result
end

# """
#     execute_to_scratch!(evaluator::AbstractEvaluator, scratch::Vector{Float64}, 
#                        scratch_start::Int, scratch_end::Int, data::NamedTuple, row_idx::Int)

# """
# function execute_to_scratch!(evaluator::AbstractEvaluator, scratch::Vector{Float64}, 
#                              scratch_start::Int, scratch_end::Int, data::NamedTuple, row_idx::Int)
#     if evaluator isa ConstantEvaluator
#         @inbounds scratch[scratch_start] = evaluator.value

#     elseif evaluator isa ContinuousEvaluator
#         @inbounds scratch[scratch_start] = Float64(data[evaluator.column][row_idx])

#     elseif evaluator isa CategoricalEvaluator
#         @inbounds cat_val = data[evaluator.column][row_idx]
#         level_idx = extract_categorical_level_fast(cat_val, evaluator.n_levels)

#         contrast_matrix = evaluator.contrast_matrix
#         n_contrasts = scratch_end - scratch_start + 1

#         @inbounds for j in 1:n_contrasts
#             scratch[scratch_start + j - 1] = contrast_matrix[level_idx, j]
#         end

#     elseif evaluator isa FunctionEvaluator
#         # ZERO-ALLOC FunctionEvaluator: handle up to 3 args inline
#         n_args = length(evaluator.arg_evaluators)

#         if n_args == 1
#             child = evaluator.arg_evaluators[1]
#             val = if is_direct_evaluatable(child)
#                 child isa ConstantEvaluator ? child.value : Float64(data[child.column][row_idx])
#             else
#                 r = evaluator.arg_scratch_map[1]
#                 execute_to_scratch!(child, scratch, first(r), last(r), data, row_idx)
#                 scratch[first(r)]
#             end
#             result = apply_function_safe(evaluator.func, val)

#         elseif n_args == 2
#             c1, c2 = evaluator.arg_evaluators
#             v1 = if is_direct_evaluatable(c1)
#                 c1 isa ConstantEvaluator ? c1.value : Float64(data[c1.column][row_idx])
#             else
#                 r1 = evaluator.arg_scratch_map[1]
#                 execute_to_scratch!(c1, scratch, first(r1), last(r1), data, row_idx)
#                 scratch[first(r1)]
#             end
#             v2 = if is_direct_evaluatable(c2)
#                 c2 isa ConstantEvaluator ? c2.value : Float64(data[c2.column][row_idx])
#             else
#                 r2 = evaluator.arg_scratch_map[2]
#                 execute_to_scratch!(c2, scratch, first(r2), last(r2), data, row_idx)
#                 scratch[first(r2)]
#             end
#             result = apply_function_safe(evaluator.func, v1, v2)

#         elseif n_args == 3
#             c1, c2, c3 = evaluator.arg_evaluators
#             v1 = if is_direct_evaluatable(c1)
#                 c1 isa ConstantEvaluator ? c1.value : Float64(data[c1.column][row_idx])
#             else
#                 r1 = evaluator.arg_scratch_map[1]
#                 execute_to_scratch!(c1, scratch, first(r1), last(r1), data, row_idx)
#                 scratch[first(r1)]
#             end
#             v2 = if is_direct_evaluatable(c2)
#                 c2 isa ConstantEvaluator ? c2.value : Float64(data[c2.column][row_idx])
#             else
#                 r2 = evaluator.arg_scratch_map[2]
#                 execute_to_scratch!(c2, scratch, first(r2), last(r2), data, row_idx)
#                 scratch[first(r2)]
#             end
#             v3 = if is_direct_evaluatable(c3)
#                 c3 isa ConstantEvaluator ? c3.value : Float64(data[c3.column][row_idx])
#             else
#                 r3 = evaluator.arg_scratch_map[3]
#                 execute_to_scratch!(c3, scratch, first(r3), last(r3), data, row_idx)
#                 scratch[first(r3)]
#             end
#             result = apply_function_safe(evaluator.func, v1, v2, v3)

#         else
#             error("FunctionEvaluator with $n_args args not supported (use ParametricFunctionEvaluator for >3)")
#         end
#         @inbounds scratch[scratch_start] = result

#     elseif evaluator isa ParametricFunctionEvaluator
#         # Generic zero-alloc for >3 args
#         # Build an NTuple of argument values without heap allocations
#         N = length(evaluator.arg_evaluators)
#         vals = ntuple(i -> begin
#             child = evaluator.arg_evaluators[i]
#             if is_direct_evaluatable(child)
#                 child isa ConstantEvaluator ? child.value : Float64(data[child.column][row_idx])
#             else
#                 r = evaluator.arg_scratch_map[i]
#                 execute_to_scratch!(child, scratch, first(r), last(r), data, row_idx)
#                 scratch[first(r)]
#             end
#         end, N)
#         result = apply_function_safe(evaluator.func, vals...)
#         @inbounds scratch[scratch_start] = result

#     elseif evaluator isa InteractionEvaluator
#         # General N-way interaction handling
#         n_components = length(evaluator.components)
#         @inbounds for i in 1:n_components
#             comp_range = evaluator.component_scratch_map[i]
#             execute_to_scratch!(evaluator.components[i], scratch, first(comp_range), last(comp_range), data, row_idx)
#         end
#         apply_kronecker_to_scratch_range!(
#             evaluator.kronecker_pattern,
#             evaluator.component_scratch_map,
#             scratch,
#             scratch_start,
#             scratch_end
#         )

#     elseif evaluator isa ZScoreEvaluator
#         # Z-score transformation
#         underlying_range = evaluator.underlying_scratch_map
#         execute_to_scratch!(evaluator.underlying, scratch, first(underlying_range), last(underlying_range), data, row_idx)
#         center = evaluator.center
#         scale = evaluator.scale
#         n_outputs = scratch_end - scratch_start + 1
#         @inbounds for i in 1:n_outputs
#             input_val = scratch[first(underlying_range) + i - 1]
#             transformed_val = if scale ≈ 0.0
#                 input_val ≈ center ? 0.0 : (input_val > center ? Inf : -Inf)
#             else
#                 (input_val - center) / scale
#             end
#             scratch[scratch_start + i - 1] = transformed_val
#         end

#     elseif evaluator isa ScaledEvaluator
#         # Scaled evaluation
#         underlying_range = evaluator.underlying_scratch_map
#         execute_to_scratch!(evaluator.evaluator, scratch, first(underlying_range), last(underlying_range), data, row_idx)
#         scale_factor = evaluator.scale_factor
#         n_outputs = scratch_end - scratch_start + 1
#         @inbounds for i in 1:n_outputs
#             input_val = scratch[first(underlying_range) + i - 1]
#             scratch[scratch_start + i - 1] = scale_factor * input_val
#         end

#     elseif evaluator isa ProductEvaluator
#         # Product evaluation
#         n_components = length(evaluator.components)
#         @inbounds for i in 1:n_components
#             comp_range = evaluator.component_scratch_map[i]
#             execute_to_scratch!(evaluator.components[i], scratch, first(comp_range), last(comp_range), data, row_idx)
#         end
#         product = 1.0
#         @inbounds for i in 1:n_components
#             comp_range = evaluator.component_scratch_map[i]
#             product *= scratch[first(comp_range)]
#         end
#         @inbounds scratch[scratch_start] = product

#     elseif evaluator isa CombinedEvaluator
#         # CombinedEvaluator concatenates multiple child evaluators
#         for i in eachindex(evaluator.evaluators)
#             child = evaluator.evaluators[i]
#             r     = evaluator.scratch_maps[i]
#             execute_to_scratch!(child, scratch, first(r), last(r), data, row_idx)
#         end

#     else
#         error("Unknown evaluator type: $(typeof(evaluator))")
#     end
#     return nothing
# end

@inline function execute_to_scratch!(
    evaluator::ConstantEvaluator, scratch, start, end_, data, row_idx)
    @inbounds scratch[start] = evaluator.value
    return nothing
 end

@inline function execute_to_scratch!(
    evaluator::ContinuousEvaluator, scratch, start, end_, data, row_idx)
    col = evaluator.column  # Cache field access
    @inbounds scratch[start] = Float64(data[col][row_idx])
    return nothing
end

@inline function execute_to_scratch!(
    evaluator::CategoricalEvaluator,
    scratch::Vector{Float64},
    scratch_start::Int,
    scratch_end::Int,
    data::NamedTuple,
    row_idx::Int
)
    # Cache field accesses
    level_codes = evaluator.level_codes
    cm = evaluator.contrast_matrix
    n_levels = evaluator.n_levels
    
    @inbounds begin
        # Direct level code access - no levelcode() call!
        lvl = level_codes[row_idx]
        # Clamp to valid range
        lvl = lvl < 1 ? 1 : (lvl > n_levels ? n_levels : lvl)
        
        n = scratch_end - scratch_start + 1
        for j in 1:n
            scratch[scratch_start + j - 1] = cm[lvl, j]
        end
    end
    return nothing
end

@inline function execute_to_scratch!(
   evaluator::FunctionEvaluator,
   scratch::Vector{Float64},
   scratch_start::Int,
   scratch_end::Int,
   data::NamedTuple,
   row_idx::Int
)
   # Cache all field accesses at the top
   arg_evaluators = evaluator.arg_evaluators
   arg_scratch_map = evaluator.arg_scratch_map
   func = evaluator.func
   
   n_args = length(arg_evaluators)

   if n_args == 1
       c1 = arg_evaluators[1]
       v1 = c1 isa ConstantEvaluator ? c1.value :
            (c1 isa ContinuousEvaluator ? Float64(data[c1.column][row_idx]) :
              begin
                r1 = arg_scratch_map[1]
                execute_to_scratch!(c1, scratch, first(r1), last(r1), data, row_idx)
                scratch[first(r1)]
              end)
       result = apply_function_safe(func, v1)

   elseif n_args == 2
       c1, c2 = arg_evaluators
       # first arg
       v1 = c1 isa ConstantEvaluator ? c1.value :
            (c1 isa ContinuousEvaluator ? Float64(data[c1.column][row_idx]) :
              begin
                r1 = arg_scratch_map[1]
                execute_to_scratch!(c1, scratch, first(r1), last(r1), data, row_idx)
                scratch[first(r1)]
              end)
       # second arg
       v2 = c2 isa ConstantEvaluator ? c2.value :
            (c2 isa ContinuousEvaluator ? Float64(data[c2.column][row_idx]) :
              begin
                r2 = arg_scratch_map[2]
                execute_to_scratch!(c2, scratch, first(r2), last(r2), data, row_idx)
                scratch[first(r2)]
              end)
       result = apply_function_safe(func, v1, v2)

   elseif n_args == 3
       c1, c2, c3 = arg_evaluators

       # v1
       v1 = if c1 isa ConstantEvaluator
           c1.value
       elseif c1 isa ContinuousEvaluator
           Float64(data[c1.column][row_idx])
       else
           r1 = arg_scratch_map[1]
           execute_to_scratch!(c1, scratch, first(r1), last(r1), data, row_idx)
           scratch[first(r1)]
       end

       # v2
       v2 = if c2 isa ConstantEvaluator
           c2.value
       elseif c2 isa ContinuousEvaluator
           Float64(data[c2.column][row_idx])
       else
           r2 = arg_scratch_map[2]
           execute_to_scratch!(c2, scratch, first(r2), last(r2), data, row_idx)
           scratch[first(r2)]
       end

       # v3
       v3 = if c3 isa ConstantEvaluator
           c3.value
       elseif c3 isa ContinuousEvaluator
           Float64(data[c3.column][row_idx])
       else
           r3 = arg_scratch_map[3]
           execute_to_scratch!(c3, scratch, first(r3), last(r3), data, row_idx)
           scratch[first(r3)]
       end

       result = apply_function_safe(func, v1, v2, v3)

   else
       # by construction, you never get here (n_args ≤ 3)
       error("FunctionEvaluator with $n_args args – should be handled by ParametricFunctionEvaluator")
   end

   @inbounds scratch[scratch_start] = result
   return nothing
end
=#