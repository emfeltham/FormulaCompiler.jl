# execution.jl

###############################################################################
# ENSURE ALL EVALUATOR TYPES HAVE execute_self_contained! METHODS
###############################################################################

# """
#     execute_zscore_self_contained!(evaluator::ZScoreEvaluator, scratch::Vector{Float64},
#                                   output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)

# UPDATED: Remove view allocation.
# """
# function execute_zscore_self_contained!(evaluator::ZScoreEvaluator, scratch::Vector{Float64},
#                                        output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)
    
#     # Evaluate underlying into scratch space - FIXED: no view allocation
#     underlying_start = first(evaluator.underlying_scratch_map)
#     underlying_end = last(evaluator.underlying_scratch_map)
#     execute_to_scratch!(evaluator.underlying, scratch, underlying_start, underlying_end, data, row_idx)
    
#     # Apply Z-score transformation: (x - center) / scale
#     center = evaluator.center
#     scale = evaluator.scale
    
#     positions = evaluator.positions
#     @inbounds for i in 1:length(positions)
#         pos        = positions[i]
#         scratch_val = scratch[underlying_start + i - 1]
#         if scale ≈ 0.0
#             output[pos] = scratch_val ≈ center ? 0.0 : (scratch_val > center ? Inf : -Inf)
#         else
#             output[pos] = (scratch_val - center) / scale
#         end
#     end
    
#     return nothing
# end

# """
#     execute_scaled_self_contained!(evaluator::ScaledEvaluator, scratch::Vector{Float64},
#                                   output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)

# UPDATED: Remove view allocation.
# """
# function execute_scaled_self_contained!(evaluator::ScaledEvaluator, scratch::Vector{Float64},
#                                        output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)
    
#     # Evaluate underlying into scratch space - FIXED: no view allocation
#     underlying_start = first(evaluator.underlying_scratch_map)
#     underlying_end = last(evaluator.underlying_scratch_map)
#     execute_to_scratch!(evaluator.evaluator, scratch, underlying_start, underlying_end, data, row_idx)
    
#     # Apply scaling: scale_factor * value
#     scale_factor = evaluator.scale_factor
    
#     positions = evaluator.positions
#     @inbounds for i in 1:length(positions)
#         pos        = positions[i]
#         scratch_val = scratch[underlying_start + i - 1]
#         output[pos] = scale_factor * scratch_val
#     end
    
#     return nothing
# end

# """
#     execute_product_self_contained!(evaluator::ProductEvaluator, scratch::Vector{Float64},
#                                    output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)

# UPDATED: Remove dynamic vector allocation.
# """
# function execute_product_self_contained!(evaluator::ProductEvaluator, scratch::Vector{Float64},
#                                         output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)
    
#     # Evaluate each component into its assigned scratch space
#     components = evaluator.components
#     component_scratch_map = evaluator.component_scratch_map
#     n_components = length(components)
#     @inbounds for i in 1:n_components
#         component_range = component_scratch_map[i]
#         component_start = first(component_range)
#         component_end = last(component_range)
#         execute_to_scratch!(components[i], scratch, component_start, component_end, data, row_idx)
#     end
    
#     # Compute product - FIXED: no dynamic vector allocation
#     product = 1.0
#     @inbounds for i in 1:n_components
#         component_range = evaluator.component_scratch_map[i]
#         val = scratch[first(component_range)]  # Products assume scalar components
#         product *= val
#     end
    
#     @inbounds output[evaluator.position] = product
    
#     return nothing
# end

# """
#     is_direct_evaluatable(evaluator::AbstractEvaluator) -> Bool

# Check if evaluator can be evaluated directly without scratch space.
# """
# function is_direct_evaluatable(evaluator::AbstractEvaluator)
#     return evaluator isa ConstantEvaluator || evaluator isa ContinuousEvaluator
# end

###############################################################################
# HELPER FUNCTIONS FOR EXECUTION PLAN CREATION
###############################################################################

# """
#     create_function_operations_from_scratch_map(evaluator::FunctionEvaluator) -> Vector{FunctionOp}

# Create function operations using evaluator's scratch mapping.
# """
# function create_function_operations_from_scratch_map(evaluator::FunctionEvaluator)
#     # For now, return simple operation - complex decomposition would go here
#     operations = FunctionOp[]
    
#     # Create operation that reads from scratch and writes to output
#     input_sources = InputSource[]
#     for scratch_range in evaluator.arg_scratch_map
#         if length(scratch_range) == 1
#             push!(input_sources, ScratchSource(first(scratch_range)))
#         else
#             error("Multi-output arguments not yet supported in function operations")
#         end
#     end
    
#     output_dest = OutputPosition(evaluator.position)
#     op = FunctionOp(evaluator.func, input_sources, output_dest)
#     push!(operations, op)
    
#     return operations
# end
