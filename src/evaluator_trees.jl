###############################################################################
# ROOT EVALUATOR ACCESS FUNCTIONS
###############################################################################

"""
    extract_root_evaluator(compiled_formula::CompiledFormula) -> AbstractEvaluator

Extract the root evaluator from a compiled formula.
Now this is trivial since we store it directly!

# Example
```julia
compiled = compile_formula(model)
evaluator = extract_root_evaluator(compiled)
```
"""
function extract_root_evaluator(compiled_formula::CompiledFormula)
    return compiled_formula.root_evaluator
end

"""
    get_evaluator_tree(compiled_formula::CompiledFormula) -> AbstractEvaluator

Alias for extract_root_evaluator with more descriptive name.
"""
function get_evaluator_tree(compiled_formula::CompiledFormula)
    return compiled_formula.root_evaluator
end

"""
    has_evaluator_access(compiled_formula::CompiledFormula) -> Bool

Check if a CompiledFormula provides access to its evaluator tree.
With the new structure, this is always true.
"""
function has_evaluator_access(compiled_formula::CompiledFormula)
    return true  # Always true with new structure
end

###############################################################################
# EVALUATOR TREE ANALYSIS FUNCTIONS
###############################################################################

"""
    count_evaluator_nodes(compiled_formula::CompiledFormula) -> Int

Count the total number of nodes in the evaluator tree.
Useful for analyzing formula complexity.
"""
function count_evaluator_nodes(compiled_formula::CompiledFormula)
    return count_nodes_recursive(compiled_formula.root_evaluator)
end

function count_nodes_recursive(evaluator::AbstractEvaluator)
    count = 1  # Count this node
    
    if evaluator isa FunctionEvaluator
        for arg_eval in evaluator.arg_evaluators
            count += count_nodes_recursive(arg_eval)
        end
    elseif evaluator isa InteractionEvaluator
        for comp_eval in evaluator.components
            count += count_nodes_recursive(comp_eval)
        end
    elseif evaluator isa CombinedEvaluator
        for sub_eval in evaluator.sub_evaluators
            count += count_nodes_recursive(sub_eval)
        end
    elseif evaluator isa ZScoreEvaluator
        count += count_nodes_recursive(evaluator.underlying)
    elseif evaluator isa ChainRuleEvaluator
        count += count_nodes_recursive(evaluator.inner_evaluator)
        count += count_nodes_recursive(evaluator.inner_derivative)
    elseif evaluator isa ProductRuleEvaluator
        count += count_nodes_recursive(evaluator.left_evaluator)
        count += count_nodes_recursive(evaluator.left_derivative)
        count += count_nodes_recursive(evaluator.right_evaluator)
        count += count_nodes_recursive(evaluator.right_derivative)
    end
    
    return count
end

"""
    get_variable_dependencies(compiled_formula::CompiledFormula) -> Vector{Symbol}

Get all variables that the formula depends on by analyzing the evaluator tree.
"""
function get_variable_dependencies(compiled_formula::CompiledFormula)
    variables = Symbol[]
    collect_variables_recursive!(variables, compiled_formula.root_evaluator)
    return unique(variables)
end

function collect_variables_recursive!(variables::Vector{Symbol}, evaluator::AbstractEvaluator)
    if evaluator isa ContinuousEvaluator
        push!(variables, evaluator.column)
    elseif evaluator isa CategoricalEvaluator
        push!(variables, evaluator.column)
    elseif evaluator isa FunctionEvaluator
        for arg_eval in evaluator.arg_evaluators
            collect_variables_recursive!(variables, arg_eval)
        end
    elseif evaluator isa InteractionEvaluator
        for comp_eval in evaluator.components
            collect_variables_recursive!(variables, comp_eval)
        end
    elseif evaluator isa CombinedEvaluator
        for sub_eval in evaluator.sub_evaluators
            collect_variables_recursive!(variables, sub_eval)
        end
    elseif evaluator isa ZScoreEvaluator
        collect_variables_recursive!(variables, evaluator.underlying)
    elseif evaluator isa ChainRuleEvaluator
        collect_variables_recursive!(variables, evaluator.inner_evaluator)
        collect_variables_recursive!(variables, evaluator.inner_derivative)
    elseif evaluator isa ProductRuleEvaluator
        collect_variables_recursive!(variables, evaluator.left_evaluator)
        collect_variables_recursive!(variables, evaluator.left_derivative)
        collect_variables_recursive!(variables, evaluator.right_evaluator)
        collect_variables_recursive!(variables, evaluator.right_derivative)
    end
end

"""
    get_evaluator_summary(compiled_formula::CompiledFormula) -> NamedTuple

Get a comprehensive summary of the evaluator tree structure.
"""
function get_evaluator_summary(compiled_formula::CompiledFormula)
    evaluator = compiled_formula.root_evaluator
    
    return (
        type = typeof(evaluator),
        total_nodes = count_evaluator_nodes(compiled_formula),
        output_width = compiled_formula.output_width,
        variables = get_variable_dependencies(compiled_formula),
        variable_count = length(get_variable_dependencies(compiled_formula)),
        complexity_score = estimate_complexity(evaluator)
    )
end

function estimate_complexity(evaluator::AbstractEvaluator)
    # Simple complexity heuristic based on evaluator types
    base_cost = if evaluator isa ConstantEvaluator
        1
    elseif evaluator isa ContinuousEvaluator
        1
    elseif evaluator isa CategoricalEvaluator
        2  # Categorical lookup is slightly more expensive
    elseif evaluator isa FunctionEvaluator
        5  # Function calls are more expensive
    elseif evaluator isa InteractionEvaluator
        10 # Interactions require multiple operations
    else
        3  # Default moderate cost
    end
    
    # Add complexity of child evaluators
    child_cost = 0
    if evaluator isa FunctionEvaluator
        child_cost = sum(estimate_complexity(arg) for arg in evaluator.arg_evaluators)
    elseif evaluator isa InteractionEvaluator
        child_cost = sum(estimate_complexity(comp) for comp in evaluator.components)
    elseif evaluator isa CombinedEvaluator
        child_cost = sum(estimate_complexity(sub) for sub in evaluator.sub_evaluators)
    elseif evaluator isa ZScoreEvaluator
        child_cost = estimate_complexity(evaluator.underlying)
    end
    
    return base_cost + child_cost
end

###############################################################################
# PRETTY PRINTING FOR EVALUATOR TREES
###############################################################################

"""
    Base.show(io::IO, evaluator::AbstractEvaluator)

Pretty print evaluator trees for debugging and introspection.
"""
function Base.show(io::IO, ::MIME"text/plain", compiled_formula::CompiledFormula)
    println(io, "CompiledFormula:")
    println(io, "  Output width: $(compiled_formula.output_width)")
    println(io, "  Variables: $(compiled_formula.column_names)")
    println(io, "  Root evaluator: $(typeof(compiled_formula.root_evaluator))")
    
    summary = get_evaluator_summary(compiled_formula)
    println(io, "  Total nodes: $(summary.total_nodes)")
    println(io, "  Complexity: $(summary.complexity_score)")
    println(io, "  Dependencies: $(summary.variables)")
end

function show_evaluator_tree(io::IO, evaluator::AbstractEvaluator, indent::Int = 0)
    prefix = "  " ^ indent
    
    if evaluator isa ConstantEvaluator
        println(io, "$(prefix)Constant($(evaluator.value))")
    elseif evaluator isa ContinuousEvaluator
        println(io, "$(prefix)Continuous(:$(evaluator.column))")
    elseif evaluator isa CategoricalEvaluator
        println(io, "$(prefix)Categorical(:$(evaluator.column), $(evaluator.n_levels) levels)")
    elseif evaluator isa FunctionEvaluator
        println(io, "$(prefix)Function($(evaluator.func)):")
        for arg in evaluator.arg_evaluators
            show_evaluator_tree(io, arg, indent + 1)
        end
    elseif evaluator isa InteractionEvaluator
        println(io, "$(prefix)Interaction($(length(evaluator.components)) components):")
        for comp in evaluator.components
            show_evaluator_tree(io, comp, indent + 1)
        end
    elseif evaluator isa CombinedEvaluator
        println(io, "$(prefix)Combined($(length(evaluator.sub_evaluators)) terms):")
        for sub in evaluator.sub_evaluators
            show_evaluator_tree(io, sub, indent + 1)
        end
    else
        println(io, "$(prefix)$(typeof(evaluator))")
    end
end

"""
    print_evaluator_tree(compiled_formula::CompiledFormula)

Print a detailed tree view of the evaluator structure.
"""
function print_evaluator_tree(compiled_formula::CompiledFormula)
    println("Evaluator Tree:")
    show_evaluator_tree(stdout, compiled_formula.root_evaluator, 0)
end

###############################################################################
# TESTING AND VALIDATION
###############################################################################

"""
    test_evaluator_storage(model, data) -> Bool

Test that evaluator storage works correctly and evaluator produces same results as compiled formula.
"""
function test_evaluator_storage(model, data)
    println("Testing evaluator storage...")
    
    # Compile formula with new storage
    compiled = compile_formula(model)
    println("✓ Compiled formula with evaluator storage")
    
    # Test that we can access the evaluator
    evaluator = extract_root_evaluator(compiled)
    println("✓ Extracted evaluator: $(typeof(evaluator))")
    
    # Test that evaluator produces same results as compiled formula
    data_nt = Tables.columntable(data)
    n_test = min(10, length(first(data_nt)))
    
    for i in 1:n_test
        # Evaluate using compiled formula
        compiled_result = Vector{Float64}(undef, length(compiled))
        compiled(compiled_result, data_nt, i)
        
        # Evaluate using raw evaluator
        evaluator_result = Vector{Float64}(undef, output_width(evaluator))
        evaluate!(evaluator, evaluator_result, data_nt, i, 1)
        
        # Compare results
        if !isapprox(compiled_result, evaluator_result, rtol=1e-12)
            println("✗ Results don't match at row $i")
            println("  Compiled: $compiled_result")
            println("  Evaluator: $evaluator_result")
            return false
        end
    end
    
    println("✓ Evaluator produces identical results to compiled formula")
    
    # Test analysis functions
    summary = get_evaluator_summary(compiled)
    println("✓ Generated evaluator summary: $summary")
    
    dependencies = get_variable_dependencies(compiled)
    println("✓ Found variable dependencies: $dependencies")
    
    node_count = count_evaluator_nodes(compiled)
    println("✓ Counted evaluator nodes: $node_count")
    
    return true
end
