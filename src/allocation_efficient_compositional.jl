# allocation_efficient_compositional.jl
# Minimize allocations even during compilation

###############################################################################
# Pre-allocated Compilation Structures
###############################################################################

"""
Compilation context with pre-allocated structures to minimize allocation during compilation
"""
mutable struct CompilationContext
    # Pre-allocated vectors (reused across multiple compilations)
    instruction_buffer::Vector{String}
    node_buffer::Vector{ExpressionNode}
    column_buffer::Vector{Symbol}
    position_counter::Ref{Int}
    node_positions::Dict{ExpressionNode, Int}
    
    # Constructor with reasonable initial sizes
    function CompilationContext()
        new(
            String[],           # Will grow as needed
            ExpressionNode[],   # Will grow as needed  
            Symbol[],           # Will grow as needed
            Ref(1),
            Dict{ExpressionNode, Int}()
        )
    end
end

# Global context to reuse across compilations
const GLOBAL_COMPILATION_CONTEXT = CompilationContext()

function reset_context!(ctx::CompilationContext)
    empty!(ctx.instruction_buffer)
    empty!(ctx.node_buffer)
    empty!(ctx.column_buffer)
    ctx.position_counter[] = 1
    empty!(ctx.node_positions)
end

###############################################################################
# Allocation-Efficient Parsing with Fixed-Size Returns
###############################################################################

"""
Parse terms with minimal allocation by using a more efficient return strategy
"""
function parse_term_efficient!(ctx::CompilationContext, term::AbstractTerm)
    # Clear node buffer for this term
    start_length = length(ctx.node_buffer)
    
    parse_term_into_buffer!(ctx.node_buffer, term)
    
    # Return view into buffer instead of copying
    end_length = length(ctx.node_buffer)
    return view(ctx.node_buffer, (start_length+1):end_length)
end

function parse_term_into_buffer!(buffer::Vector{ExpressionNode}, term::AbstractTerm)
    if term isa InterceptTerm
        hasintercept(term) && push!(buffer, ConstantNode(1.0))
        
    elseif term isa ConstantTerm
        push!(buffer, ConstantNode(Float64(term.n)))
        
    elseif term isa Union{ContinuousTerm, Term}
        push!(buffer, ColumnNode(term.sym))
        
    elseif term isa CategoricalTerm
        contrast_matrix = Matrix{Float64}(term.contrasts.matrix)
        n_cols = size(contrast_matrix, 2)
        for j in 1:n_cols
            push!(buffer, CategoricalNode(term.sym, contrast_matrix, j))
        end
        
    elseif term isa ZScoredTerm
        # Parse underlying term first
        underlying_start = length(buffer)
        parse_term_into_buffer!(buffer, term.term)
        underlying_end = length(buffer)
        
        # Extract standardization parameters
        center = term.center isa Number ? Float64(term.center) : Float64(term.center[1])
        scale = term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
        
        # Replace underlying nodes with ZScore wrapped versions
        for i in (underlying_start+1):underlying_end
            underlying_node = buffer[i]
            buffer[i] = ZScoreNode(underlying_node, center, scale)
        end
        
    elseif term isa FunctionTerm
        parse_function_into_buffer!(buffer, term)
        
    elseif term isa InteractionTerm
        parse_interaction_into_buffer!(buffer, term)
        
    else
        error("Unsupported term type: $(typeof(term))")
    end
end

function parse_function_into_buffer!(buffer::Vector{ExpressionNode}, term::FunctionTerm)
    func = term.f
    args = term.args
    
    if length(args) == 1
        # Unary function
        arg_start = length(buffer)
        parse_term_into_buffer!(buffer, args[1])
        arg_end = length(buffer)
        
        # Wrap each argument node with the function
        for i in (arg_start+1):arg_end
            arg_node = buffer[i]
            buffer[i] = UnaryOpNode(func, arg_node)
        end
        
    elseif length(args) == 2
        # Binary function
        left_start = length(buffer)
        parse_term_into_buffer!(buffer, args[1])
        left_end = length(buffer)
        
        right_start = length(buffer)
        parse_term_into_buffer!(buffer, args[2])
        right_end = length(buffer)
        
        # Should have single nodes for binary operations
        if (left_end - left_start) == 1 && (right_end - right_start) == 1
            left_node = buffer[left_start + 1]
            right_node = buffer[right_start + 1]
            
            # Replace with binary operation
            resize!(buffer, left_start + 1)
            buffer[left_start + 1] = BinaryOpNode(func, left_node, right_node)
        else
            error("Binary functions with multi-valued arguments not supported: $term")
        end
        
    else
        error("Functions with $(length(args)) arguments not yet supported: $term")
    end
end

function parse_interaction_into_buffer!(buffer::Vector{ExpressionNode}, term::InteractionTerm)
    # Parse all components into separate sections of the buffer
    component_ranges = Tuple{Int, Int}[]  # (start, end) for each component
    
    for component in term.terms
        start_idx = length(buffer)
        parse_term_into_buffer!(buffer, component)
        end_idx = length(buffer)
        push!(component_ranges, (start_idx + 1, end_idx))
    end
    
    # Generate combinations without allocating intermediate vectors
    interaction_start = length(buffer)
    generate_combinations_into_buffer!(buffer, component_ranges)
    
    # Remove the original component nodes, keep only interactions
    # (This is tricky - for now, just keep everything and mark interactions specially)
end

function generate_combinations_into_buffer!(buffer::Vector{ExpressionNode}, component_ranges::Vector{Tuple{Int, Int}})
    # For now, implement simple 2-component case efficiently
    if length(component_ranges) == 2
        range1_start, range1_end = component_ranges[1]
        range2_start, range2_end = component_ranges[2]
        
        for i in range1_start:range1_end
            for j in range2_start:range2_end
                node1 = buffer[i]
                node2 = buffer[j]
                push!(buffer, InteractionNode([node1, node2]))
            end
        end
    else
        # General case - can be optimized further
        generate_combinations_general!(buffer, component_ranges)
    end
end

function generate_combinations_general!(buffer::Vector{ExpressionNode}, component_ranges::Vector{Tuple{Int, Int}})
    # Recursive combination generation without intermediate allocation
    combinations = Vector{ExpressionNode}[]  # This allocates, but only temporarily
    
    function generate_recursive!(current_combination, range_idx)
        if range_idx > length(component_ranges)
            push!(buffer, InteractionNode(copy(current_combination)))
        else
            start_idx, end_idx = component_ranges[range_idx]
            for i in start_idx:end_idx
                push!(current_combination, buffer[i])
                generate_recursive!(current_combination, range_idx + 1)
                pop!(current_combination)
            end
        end
    end
    
    generate_recursive!(ExpressionNode[], 1)
end

###############################################################################
# Allocation-Efficient Code Generation
###############################################################################

function compile_expression_tree_efficient!(ctx::CompilationContext, expression_nodes)
    # Reset instruction buffer
    empty!(ctx.instruction_buffer)
    empty!(ctx.node_positions)
    ctx.position_counter[] = 1
    
    # Compile each node
    for i in 1:length(expression_nodes)
        node = expression_nodes[i]
        compile_node_efficient!(ctx, node)
    end
    
    return ctx.instruction_buffer, ctx.position_counter[] - 1
end

function compile_node_efficient!(ctx::CompilationContext, node::ExpressionNode)
    # Check if already compiled
    if haskey(ctx.node_positions, node)
        return ctx.node_positions[node]
    end
    
    pos = ctx.position_counter[]
    ctx.position_counter[] += 1
    
    # Generate instruction based on node type
    if node isa ConstantNode
        push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = $(node.value)")
        
    elseif node isa ColumnNode
        push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = Float64(data.$(node.column)[row_idx])")
        push!(ctx.column_buffer, node.column)
        
    elseif node isa UnaryOpNode
        child_pos = compile_node_efficient!(ctx, node.child)
        compile_unary_op_efficient!(ctx.instruction_buffer, node.op, pos, child_pos)
        
    elseif node isa BinaryOpNode
        left_pos = compile_node_efficient!(ctx, node.left)
        right_pos = compile_node_efficient!(ctx, node.right)
        compile_binary_op_efficient!(ctx.instruction_buffer, node.op, pos, left_pos, right_pos)
        
    elseif node isa InteractionNode
        child_positions = [compile_node_efficient!(ctx, child) for child in node.children]
        compile_interaction_efficient!(ctx.instruction_buffer, pos, child_positions)
        
    elseif node isa ZScoreNode
        child_pos = compile_node_efficient!(ctx, node.child)
        compile_zscore_efficient!(ctx.instruction_buffer, pos, child_pos, node.center, node.scale)
        
    elseif node isa CategoricalNode
        compile_categorical_efficient!(ctx.instruction_buffer, node, pos)
        push!(ctx.column_buffer, node.column)
        
    else
        error("Unknown node type: $(typeof(node))")
    end
    
    ctx.node_positions[node] = pos
    return pos
end

# Efficient instruction generation functions (same logic as before, but operating on pre-allocated buffer)
function compile_unary_op_efficient!(instructions, op::Function, pos, child_pos)
    if op === log
        push!(instructions, "@inbounds val = row_vec[$child_pos]")
        push!(instructions, "@inbounds row_vec[$pos] = val > 0.0 ? log(val) : log(abs(val) + 1e-16)")
    elseif op === (^) && child_pos == pos - 1  # Special case: x^2 where the constant is known
        # This would need more sophisticated analysis to detect constants
        push!(instructions, "@inbounds row_vec[$pos] = $op(row_vec[$child_pos])")
    else
        push!(instructions, "@inbounds row_vec[$pos] = $op(row_vec[$child_pos])")
    end
end

function compile_binary_op_efficient!(instructions, op::Function, pos, left_pos, right_pos)
    if op === (+)
        push!(instructions, "@inbounds row_vec[$pos] = row_vec[$left_pos] + row_vec[$right_pos]")
    elseif op === (*)
        push!(instructions, "@inbounds row_vec[$pos] = row_vec[$left_pos] * row_vec[$right_pos]")
    elseif op === (^)
        push!(instructions, "@inbounds row_vec[$pos] = row_vec[$left_pos]^row_vec[$right_pos]")
    else
        push!(instructions, "@inbounds row_vec[$pos] = $op(row_vec[$left_pos], row_vec[$right_pos])")
    end
end

function compile_interaction_efficient!(instructions, pos, child_positions)
    if length(child_positions) == 2
        push!(instructions, "@inbounds row_vec[$pos] = row_vec[$(child_positions[1])] * row_vec[$(child_positions[2])]")
    else
        push!(instructions, "@inbounds product = row_vec[$(child_positions[1])]")
        for i in 2:length(child_positions)
            push!(instructions, "@inbounds product *= row_vec[$(child_positions[i])]")
        end
        push!(instructions, "@inbounds row_vec[$pos] = product")
    end
end

function compile_zscore_efficient!(instructions, pos, child_pos, center, scale)
    push!(instructions, "@inbounds val = row_vec[$child_pos]")
    push!(instructions, "@inbounds row_vec[$pos] = (val - $center) / $scale")
end

function compile_categorical_efficient!(instructions, node::CategoricalNode, pos)
    col = node.column
    level_idx = node.level_index
    
    values = [node.contrast_matrix[i, level_idx] for i in 1:size(node.contrast_matrix, 1)]
    
    push!(instructions, "@inbounds cat_val = data.$(col)[row_idx]")
    push!(instructions, "@inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
    
    if length(values) <= 3
        condition_chain = join([
            "level_code == $i ? $(values[i])" for i in 1:length(values)-1
        ], " : ") * " : $(values[end])"
        push!(instructions, "@inbounds row_vec[$pos] = $condition_chain")
    else
        # Generate lookup function
        lookup_name = "lookup_$(col)_$(level_idx)_$(hash(values))"
        generate_lookup_function(lookup_name, values)
        push!(instructions, "@inbounds row_vec[$pos] = $lookup_name(level_code)")
    end
end

###############################################################################
# Main Efficient Interface
###############################################################################

"""
    compile_formula_compositional_efficient(model)

Compositional compiler with minimized allocation during compilation
"""
function compile_formula_compositional_efficient(model)
    ctx = GLOBAL_COMPILATION_CONTEXT
    reset_context!(ctx)
    
    rhs = fixed_effects_form(model).rhs
    
    # Parse all terms efficiently
    for term in rhs.terms
        parse_term_into_buffer!(ctx.node_buffer, term)
    end
    
    # Compile to instructions
    instructions, output_width = compile_expression_tree_efficient!(ctx, ctx.node_buffer)
    
    # Generate function
    formula_hash = hash(string(rhs))
    func_name = generate_efficient_function(instructions, formula_hash)
    
    return func_name, output_width, unique(ctx.column_buffer)
end

function generate_efficient_function(instructions, formula_hash)
    func_name = Symbol("efficient_compositional_$(abs(formula_hash))")
    
    func_code = """
    function $func_name(row_vec, data, row_idx)
        $(join(instructions, "\n        "))
        return row_vec
    end
    """
    
    eval(Meta.parse(func_code))
    return func_name
end

#######

export compile_formula_compositional_efficient, CompilationContext

function compile_to_instructions(model)
    # Use compositional compiler instead of limited parsing
    func_name, output_width, column_names = compile_formula_compositional_efficient(model)
    
    # For compatibility with @generated interface, we need to return instructions
    # For now, return dummy instructions - the real work is done by func_name
    dummy_instructions = [LoadConstant(1, 1.0)]  # Placeholder
    
    return dummy_instructions, column_names, output_width
end

# UPDATE the modelrow! @generated function to use the compositional function:
@generated function modelrow!(row_vec, ::Val{formula_hash}, data, row_idx) where formula_hash
    # Get the compositional function name from cache or rebuild
    if haskey(FORMULA_CACHE, formula_hash)
        _, column_names, output_width = FORMULA_CACHE[formula_hash]
        
        # The compositional function should already be defined
        func_name = Symbol("efficient_compositional_$(abs(formula_hash))")
        
        return quote
            $func_name(row_vec, data, row_idx)
        end
    else
        error("Formula not found in cache for hash $formula_hash")
    end
end

function compile_formula_generated(model)
    # Use compositional compiler
    func_name, output_width, column_names = compile_formula_compositional_efficient(model)
    
    # Register in cache for @generated dispatch
    formula_hash = hash(string(fixed_effects_form(model).rhs))
    dummy_instructions = [LoadConstant(1, 1.0)]  # Placeholder for compatibility
    FORMULA_CACHE[formula_hash] = (dummy_instructions, column_names, output_width)
    
    return (Val(formula_hash), output_width, column_names)
end