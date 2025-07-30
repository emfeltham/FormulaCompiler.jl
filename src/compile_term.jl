# compile_term.jl

"""
    compile_function_term(term::FunctionTerm, start_position::Int,
                                        scratch_allocator::ScratchAllocator)

Compile function term with argument scratch space allocation.
"""
function compile_function_term(term::FunctionTerm, start_position::Int,
                              scratch_allocator::ScratchAllocator)
    
    # Compile arguments with proper scratch positions
    arg_evaluators = AbstractEvaluator[]
    arg_scratch_map = UnitRange{Int}[]
    
    for arg in term.args
        # FIXED: Allocate scratch space first, then compile with those positions
        # arg_width = width(arg)  # Use StatsModels width() before compilation
        arg_width = 1 # ALWAYS = 1?
        arg_scratch = allocate_scratch!(scratch_allocator, arg_width)
        
        # FIXED: Compile argument with its actual scratch positions
        arg_start_pos = first(arg_scratch)  # Start of its scratch space
        arg_eval = compile_term(arg, arg_start_pos, scratch_allocator)
        
        push!(arg_evaluators, arg_eval)
        push!(arg_scratch_map, arg_scratch)
    end
    
    # Collect all scratch positions used
    all_scratch = Int[]
    for range in arg_scratch_map
        append!(all_scratch, collect(range))
    end
    
    evaluator = FunctionEvaluator(
        term.f,
        arg_evaluators,
        start_position,  # This function's output position
        all_scratch,
        arg_scratch_map
    )
    
    return evaluator
end

"""
    compile_interaction_term(term::InteractionTerm, start_position::Int,
                            scratch_allocator::ScratchAllocator)

Compile interaction term with component scratch space allocation.
Updated to use parametric InteractionEvaluator{N} for type stability.
"""
function compile_interaction_term(term::InteractionTerm, start_position::Int,
                                 scratch_allocator::ScratchAllocator)
    
    # Compile components with proper scratch positions
    component_evaluators = AbstractEvaluator[]
    component_scratch_map = UnitRange{Int}[]
    component_widths = Int[]
    
    for comp in term.terms
        # FIXED: Allocate scratch space first, then compile with those positions
        comp_width = width(comp)  # Use StatsModels width() before compilation
        push!(component_widths, comp_width)
        comp_scratch = allocate_scratch!(scratch_allocator, comp_width)
        
        # FIXED: Compile component with its actual scratch positions
        comp_start_pos = first(comp_scratch)  # Start of its scratch space
        comp_eval = compile_term(comp, comp_start_pos, scratch_allocator)
        
        push!(component_evaluators, comp_eval)
        push!(component_scratch_map, comp_scratch)
    end
    
    # Calculate interaction output width and positions
    total_width = prod(component_widths)
    positions = collect(start_position:(start_position + total_width - 1))
    
    # NEW: Get the number of components for parametric type
    N = length(component_widths)
    
    # Pre-compute Kronecker product pattern with correct parametric type
    pattern = compute_kronecker_pattern(component_widths)  # Returns Vector{NTuple{N,Int}}
    
    # Collect all scratch positions used
    all_scratch = Int[]
    for range in component_scratch_map
        append!(all_scratch, collect(range))
    end
    
    # Create parametric InteractionEvaluator{N} instance
    evaluator = InteractionEvaluator{N}(
        component_evaluators,
        total_width,
        positions,
        all_scratch,
        component_scratch_map,
        pattern  # This is now Vector{NTuple{N,Int}}
    )
    
    return evaluator
end

"""
    compile_zscore_term(term::ZScoredTerm, start_position::Int,
                                      scratch_allocator::ScratchAllocator)

Compile Z-score term with underlying scratch space allocation.
"""
function compile_zscore_term(term::ZScoredTerm, start_position::Int,
                            scratch_allocator::ScratchAllocator)
    
    # FIXED: Allocate scratch space first, then compile with those positions
    underlying_width = width(term.term)  # Use StatsModels width()
    underlying_scratch = allocate_scratch!(scratch_allocator, underlying_width)
    
    # FIXED: Compile underlying term with its actual scratch positions
    underlying_start_pos = first(underlying_scratch)
    underlying_eval = compile_term(term.term, underlying_start_pos, scratch_allocator)
    
    # Z-score output goes to model matrix positions
    positions = collect(start_position:(start_position + underlying_width - 1))
    
    center = term.center isa Number ? Float64(term.center) : Float64(term.center[1])
    scale = term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
    
    evaluator = ZScoreEvaluator(
        underlying_eval,
        center,
        scale,
        positions,
        collect(underlying_scratch),
        underlying_scratch
    )
    
    return evaluator
end

"""
    compile_matrix_term(term::MatrixTerm, start_position::Int,
                                      scratch_allocator::ScratchAllocator)

Compile matrix term with shared scratch space allocation.
"""
function compile_matrix_term(
    term::MatrixTerm, 
    start_position::Int,
    scratch_allocator::ScratchAllocator,
    categorical_levels::Union{Dict{Symbol, Vector{Int}}, Nothing}
)
    sub_evaluators = AbstractEvaluator[]
    current_pos = start_position
    max_scratch = 0
    
    for sub_term in term.terms
        if width(sub_term) > 0
            sub_eval = compile_term(sub_term, current_pos, scratch_allocator, categorical_levels)
            next_pos = current_pos + output_width(sub_eval)
            push!(sub_evaluators, sub_eval)
            current_pos = next_pos
            
            sub_scratch = max_scratch_needed(sub_eval)
            max_scratch = max(max_scratch, sub_scratch)
        end
    end
    
    total_width = current_pos - start_position
    
    # FIXED: Create precomputed operations
    constant_ops = PrecomputedConstantOp[]
    continuous_ops = PrecomputedContinuousOp[]
    categorical_evals = CategoricalEvaluator[]
    function_evals = FunctionEvaluator[]
    interaction_evals = InteractionEvaluator[]
    
    for eval in sub_evaluators
        if eval isa ConstantEvaluator
            push!(constant_ops, PrecomputedConstantOp(eval.value, eval.position))
        elseif eval isa ContinuousEvaluator
            push!(continuous_ops, PrecomputedContinuousOp(eval.column, eval.position))
        elseif eval isa CategoricalEvaluator
            push!(categorical_evals, eval)
        elseif eval isa FunctionEvaluator
            push!(function_evals, eval)
        elseif eval isa InteractionEvaluator
            push!(interaction_evals, eval)
        else
            error("Unknown evaluator type in matrix term: $(typeof(eval))")
        end
    end
    
    return CombinedEvaluator(
        constant_ops,
        continuous_ops,
        categorical_evals,
        function_evals,
        interaction_evals,
        total_width,
        max_scratch
    )
end
