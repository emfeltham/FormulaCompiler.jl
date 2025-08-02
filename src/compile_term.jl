# compile_term.jl
#=
The functions in compile_term, turns each `AbstractTerm` in the model formula into a corresponding `AbstractEvaluator` so that the model matrix can be built efficiently.
- Allocates needed scratch space
- Builds position maps for terms

See .md document.
=#

"""
    compile_term(
        term::AbstractTerm, start_position::Int = 1, 
        scratch_allocator::ScratchAllocator = ScratchAllocator()
    ) -> (AbstractEvaluator, Int)

Compile term into self-contained evaluator with positions and scratch space.
"""
function compile_term(
    term::AbstractTerm, 
    start_position::Int = 1, 
    scratch_allocator::ScratchAllocator = ScratchAllocator(),
    categorical_levels::Union{Dict{Symbol, Vector{Int}}, Nothing} = nothing
)

    # Use explicit levels if provided, fall back to global context
    levels = categorical_levels
    
    if term isa InterceptTerm
        evaluator = ConstantEvaluator(hasintercept(term) ? 1.0 : 0.0, start_position)
        return evaluator

    elseif term isa ConstantTerm
        evaluator = ConstantEvaluator(Float64(term.n), start_position)
        return evaluator
        
    elseif term isa Union{ContinuousTerm, Term}
        evaluator = ContinuousEvaluator(term.sym, start_position)
        return evaluator
        
    elseif term isa CategoricalTerm

        contrast_matrix = Matrix{Float64}(term.contrasts.matrix)
        n_contrasts = size(contrast_matrix, 2)
        positions = collect(start_position:(start_position + n_contrasts - 1))
        
        # Use explicit levels if available
        level_codes = if levels !== nothing && haskey(levels, term.sym)
            levels[term.sym]
        else
            println("  WARNING: Using empty level codes for $(term.sym)!")
            Int[]  # Empty fallback
        end
        
        evaluator = CategoricalEvaluator(
            term.sym,
            contrast_matrix,
            size(contrast_matrix, 1),
            positions,
            level_codes
        )
        
        return evaluator
    elseif term isa FunctionTerm
        arg_evaluators = AbstractEvaluator[]
        arg_scratch_map = UnitRange{Int}[]
        
        for arg in term.args
            arg_width = 1 # ALWAYS 1?
            # arg_width = width(arg) # this would fail
            arg_scratch = allocate_scratch!(scratch_allocator, arg_width)
            arg_start_pos = first(arg_scratch)
            
            # Pass categorical_levels through recursive call
            arg_eval = compile_term(arg, arg_start_pos, scratch_allocator, categorical_levels)
            
            push!(arg_evaluators, arg_eval)
            push!(arg_scratch_map, arg_scratch)
        end
        
        all_scratch = Int[]
        for range in arg_scratch_map
            append!(all_scratch, collect(range))
        end
        
        return FunctionEvaluator(
            term.f,
            arg_evaluators,
            start_position,
            all_scratch,
            arg_scratch_map
        )
        
    elseif term isa InteractionTerm
        component_evaluators = AbstractEvaluator[]
        component_widths = Int[]
        
        # Compile each component independently (don't allocate scratch yet)
        for comp in term.terms
            # Use a temporary scratch allocator for component compilation
            temp_allocator = ScratchAllocator()
            comp_eval = compile_term(comp, 1, temp_allocator, categorical_levels)
            
            push!(component_evaluators, comp_eval)
            push!(component_widths, width(comp))
        end
        
        # Plan complete scratch space for all components
        component_output_ranges, component_internal_ranges, total_scratch_needed = 
            plan_interaction_scratch_space(component_evaluators)
        
        # Allocate the total scratch space from the main allocator
        if total_scratch_needed > 0
            total_scratch_range = allocate_scratch!(scratch_allocator, total_scratch_needed)
            all_scratch_positions = collect(total_scratch_range)
        else
            all_scratch_positions = Int[]
        end
        
        # Calculate final output positions and pattern
        total_width = prod(component_widths)
        positions = collect(start_position:(start_position + total_width - 1))
        N = length(component_widths)
        pattern = compute_kronecker_pattern(component_widths)
        
        return InteractionEvaluator{N}(
            component_evaluators,
            total_width,
            positions,
            all_scratch_positions,
            component_output_ranges,
            component_internal_ranges,
            total_scratch_needed,
            pattern
        )
        
    elseif term isa ZScoredTerm
        underlying_width = width(term.term)
        underlying_scratch = allocate_scratch!(scratch_allocator, underlying_width)
        underlying_start_pos = first(underlying_scratch)
        
        # Pass categorical_levels through recursive call
        underlying_eval = compile_term(term.term, underlying_start_pos, scratch_allocator, categorical_levels)
        
        positions = collect(start_position:(start_position + underlying_width - 1))
        center = term.center isa Number ? Float64(term.center) : Float64(term.center[1])
        scale = term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
        
        return ZScoreEvaluator(
            underlying_eval,
            center,
            scale,
            positions,
            collect(underlying_scratch),
            underlying_scratch
        )
    elseif term isa MatrixTerm
        # println("MATRIX") # called 2x
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
        
        # Create precomputed operations instead of storing evaluators
        constant_ops = PrecomputedConstantOp[]
        continuous_ops = PrecomputedContinuousOp[]
        categorical_evals = CategoricalEvaluator[]
        function_evals = FunctionEvaluator[]
        interaction_evals = InteractionEvaluator[]
        
        for eval in sub_evaluators
            if eval isa ConstantEvaluator
                # Precompute: extract fields at compilation time
                push!(constant_ops, PrecomputedConstantOp(eval.value, eval.position))
            elseif eval isa ContinuousEvaluator
                # Precompute: extract fields at compilation time
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
    else
        error("Unknown term type: $(typeof(term))")
    end
end
