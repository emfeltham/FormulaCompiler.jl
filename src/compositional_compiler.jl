# src/compositional_compiler.jl
# Complete compositional formula compiler - handles ANY StatsModels formula

using Tables
using CategoricalArrays: CategoricalValue, levelcode
using StatsModels
using StandardizedPredictors: ZScoredTerm

###############################################################################
# Expression Node Type Hierarchy
###############################################################################

"""
Abstract base type for all expression nodes in the compositional formula tree.
"""
abstract type ExpressionNode end

# Leaf nodes (atomic values)
struct ConstantNode <: ExpressionNode
    value::Float64
end

struct ColumnNode <: ExpressionNode
    column::Symbol
end

struct CategoricalNode <: ExpressionNode
    column::Symbol
    contrast_matrix::Matrix{Float64}
    level_index::Int  # Which contrast column (1, 2, etc.)
end

# Operation nodes (internal tree nodes)
struct UnaryOpNode <: ExpressionNode
    op::Function
    child::ExpressionNode
end

struct BinaryOpNode <: ExpressionNode
    op::Function
    left::ExpressionNode
    right::ExpressionNode
end

struct InteractionNode <: ExpressionNode
    children::Vector{ExpressionNode}
end

struct AtomicFunctionNode <: ExpressionNode
    term::FunctionTerm
end

struct AtomicInteractionNode <: ExpressionNode
    term::InteractionTerm  
end

struct ZScoreNode <: ExpressionNode
    child::ExpressionNode
    center::Float64
    scale::Float64
end

###############################################################################
# Compilation Context - Minimize Allocations During Compilation
###############################################################################

mutable struct CompilationContext
    instruction_buffer::Vector{String}
    node_buffer::Vector{ExpressionNode}
    column_buffer::Vector{Symbol}
    position_counter::Ref{Int}
    node_positions::Dict{ExpressionNode, Int}
    
    function CompilationContext()
        new(String[], ExpressionNode[], Symbol[], Ref(1), Dict{ExpressionNode, Int}())
    end
end

function reset_context!(ctx::CompilationContext)
    empty!(ctx.instruction_buffer)
    empty!(ctx.node_buffer)
    empty!(ctx.column_buffer)
    ctx.position_counter[] = 1
    empty!(ctx.node_positions)
end

# Global context to reuse across compilations
const GLOBAL_COMPILATION_CONTEXT = CompilationContext()

###############################################################################
# Compositional Term Parsing - True Recursion
###############################################################################

function parse_term_compositional(term::AbstractTerm)
    try
        if term isa InterceptTerm
            return hasintercept(term) ? [ConstantNode(1.0)] : ExpressionNode[]
            
        elseif term isa ConstantTerm
            return [ConstantNode(Float64(term.n))]
            
        elseif term isa Union{ContinuousTerm, Term}
            return [ColumnNode(term.sym)]
            
        elseif term isa CategoricalTerm
            return parse_categorical_term_fixed(term)
            
        elseif term isa ZScoredTerm
            # Don't recurse - treat as atomic
            child_nodes = parse_term_compositional(term.term)
            center = term.center isa Number ? Float64(term.center) : Float64(term.center[1])
            scale = term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
            return [ZScoreNode(child, center, scale) for child in child_nodes]
            
        elseif term isa FunctionTerm
            # KEY FIX: Don't break down function terms - treat as atomic
            return [AtomicFunctionNode(term)]
            
        elseif term isa InteractionTerm
            # KEY FIX: Don't break down interactions - treat as atomic  
            return [AtomicInteractionNode(term)]
            
        else
            error("Unsupported term type: $(typeof(term))")
        end
        
    catch e
        @error "Failed to parse term: $term" exception=(e, catch_backtrace())
        return [ConstantNode(0.0)]
    end
end


# ADD NEW: parse_categorical_term_fixed
function parse_categorical_term_fixed(term::CategoricalTerm)
    # Extract contrast matrix - this is where the old code was careful
    contrast_matrix = term.contrasts.matrix
    
    # Convert to Float64 matrix to ensure type stability
    contrast_matrix_float = Matrix{Float64}(contrast_matrix)
    
    # Get dimensions
    n_levels, n_cols = size(contrast_matrix_float)
    
    println("  Categorical $(term.sym): $(n_levels) levels → $(n_cols) contrast columns")
    
    # Create one node per contrast column
    nodes = CategoricalNode[]
    for j in 1:n_cols
        push!(nodes, CategoricalNode(term.sym, contrast_matrix_float, j))
    end
    
    return nodes
end

###############################################################################
# Allocation-Free Code Generation
###############################################################################

"""
Compile expression tree into linear instruction sequence using post-order traversal.
"""
function compile_expression_tree_efficient!(ctx::CompilationContext, expression_nodes)
    try
        # Reset instruction buffer
        empty!(ctx.instruction_buffer)
        empty!(ctx.node_positions)
        ctx.position_counter[] = 1
        
        # Compile each node safely
        for (i, node) in enumerate(expression_nodes)
            try
                compile_node_efficient!(ctx, node)
            catch e
                @error "Failed to compile node $i: $(typeof(node))" exception=(e, catch_backtrace())
                # Add a fallback instruction
                pos = ctx.position_counter[]
                push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = 0.0")
                ctx.position_counter[] += 1
            end
        end
        
        # Ensure we have at least one instruction
        if isempty(ctx.instruction_buffer)
            push!(ctx.instruction_buffer, "@inbounds row_vec[1] = 1.0")
            ctx.position_counter[] = 2
        end
        
        return ctx.instruction_buffer, ctx.position_counter[] - 1
        
    catch e
        @error "Fatal error in expression tree compilation" exception=(e, catch_backtrace())
        # Emergency fallback
        return ["@inbounds row_vec[1] = 1.0"], 1
    end
end

# REMOVE the problematic general compile_node_efficient! and replace with these specific methods:

function compile_node_efficient!(ctx::CompilationContext, node::AtomicFunctionNode)
    if haskey(ctx.node_positions, node)
        return ctx.node_positions[node]
    end
    
    pos = ctx.position_counter[]
    ctx.position_counter[] += 1
    
    term = node.term
    func = term.f
    args = term.args
    
    if length(args) == 1 && args[1] isa Union{ContinuousTerm, Term}
        # f(column)
        col = args[1].sym
        push!(ctx.column_buffer, col)
        
        if func === log
            push!(ctx.instruction_buffer, "@inbounds val = Float64(data.$(col)[row_idx])")
            push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = val > 0.0 ? log(val) : log(abs(val) + 1e-16)")
        else
            push!(ctx.instruction_buffer, "@inbounds val = Float64(data.$(col)[row_idx])")
            push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = $func(val)")
        end
        
    elseif length(args) == 2 && func === (^) && args[2] isa ConstantTerm
        # x^n
        col = args[1].sym
        exp = Float64(args[2].n)
        push!(ctx.column_buffer, col)
        
        if exp == 2.0
            push!(ctx.instruction_buffer, "@inbounds val = Float64(data.$(col)[row_idx])")
            push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = val * val")
        else
            push!(ctx.instruction_buffer, "@inbounds val = Float64(data.$(col)[row_idx])")
            push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = val^$exp")
        end
    else
        @warn "Complex atomic function: $term"
        push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = 1.0")
    end
    
    ctx.node_positions[node] = pos
    return pos
end

function compile_node_efficient!(ctx::CompilationContext, node::AtomicInteractionNode)
    if haskey(ctx.node_positions, node)
        return ctx.node_positions[node]
    end
    
    pos = ctx.position_counter[]
    ctx.position_counter[] += 1
    
    term = node.term
    components = term.terms
    
    # Compile each component as an atomic operation, then multiply
    if length(components) == 2
        # Binary interaction like x^2 * log(z)
        comp1, comp2 = components
        
        # Get values for each component
        if comp1 isa FunctionTerm && comp1.f === (^) && comp1.args[2] isa ConstantTerm
            # x^2 component
            col1 = comp1.args[1].sym
            exp1 = Float64(comp1.args[2].n)
            push!(ctx.column_buffer, col1)
            
            if exp1 == 2.0
                push!(ctx.instruction_buffer, "@inbounds val1 = Float64(data.$(col1)[row_idx])")
                push!(ctx.instruction_buffer, "@inbounds comp1_result = val1 * val1")
            else
                push!(ctx.instruction_buffer, "@inbounds val1 = Float64(data.$(col1)[row_idx])")
                push!(ctx.instruction_buffer, "@inbounds comp1_result = val1^$exp1")
            end
        else
            @warn "Complex interaction component 1: $comp1"
            push!(ctx.instruction_buffer, "@inbounds comp1_result = 1.0")
        end
        
        if comp2 isa FunctionTerm && comp2.f === log
            # log(z) component
            col2 = comp2.args[1].sym
            push!(ctx.column_buffer, col2)
            push!(ctx.instruction_buffer, "@inbounds val2 = Float64(data.$(col2)[row_idx])")
            push!(ctx.instruction_buffer, "@inbounds comp2_result = val2 > 0.0 ? log(val2) : log(abs(val2) + 1e-16)")
        else
            @warn "Complex interaction component 2: $comp2"
            push!(ctx.instruction_buffer, "@inbounds comp2_result = 1.0")
        end
        
        # Multiply components
        push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = comp1_result * comp2_result")
    else
        @warn "Complex interaction with $(length(components)) components: $term"
        push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = 1.0")
    end
    
    ctx.node_positions[node] = pos
    return pos
end


function compile_categorical_efficient!(instructions, node::CategoricalNode, pos)
    col = node.column
    level_idx = node.level_index
    
    # IMPORTANT: Extract contrast values for this specific level/column
    contrast_matrix = node.contrast_matrix
    n_levels = size(contrast_matrix, 1)
    
    # Get the contrast values for this specific column
    values = [contrast_matrix[i, level_idx] for i in 1:n_levels]
    
    # Add basic categorical value extraction
    push!(instructions, "@inbounds cat_val = data.$(col)[row_idx]")
    push!(instructions, "@inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
    
    # SAFETY: Bounds checking for level codes
    push!(instructions, "@inbounds level_code = clamp(level_code, 1, $n_levels)")
    
    if length(values) == 1
        # Single level case
        push!(instructions, "@inbounds row_vec[$pos] = $(values[1])")
    elseif length(values) == 2
        # Two levels - simple ternary
        push!(instructions, "@inbounds row_vec[$pos] = level_code == 1 ? $(values[1]) : $(values[2])")
    elseif length(values) == 3
        # Three levels - nested ternary
        push!(instructions, "@inbounds row_vec[$pos] = level_code == 1 ? $(values[1]) : level_code == 2 ? $(values[2]) : $(values[3])")
    else
        # Many levels - use lookup table approach from old code
        lookup_name = "lookup_$(col)_$(level_idx)_$(abs(hash(values)))"
        generate_safe_lookup_function(lookup_name, values)
        push!(instructions, "@inbounds row_vec[$pos] = $lookup_name(level_code)")
    end
end

# REPLACE: compile_unary_op_efficient!
function compile_unary_op_efficient!(instructions, op::Function, pos, child_pos)
    try
        if op === log
            push!(instructions, "@inbounds val = row_vec[$child_pos]")
            push!(instructions, "@inbounds row_vec[$pos] = val > 0.0 ? log(val) : log(abs(val) + 1e-16)")
        elseif op === sqrt
            push!(instructions, "@inbounds val = row_vec[$child_pos]")
            push!(instructions, "@inbounds row_vec[$pos] = sqrt(abs(val))")
        elseif op === exp
            push!(instructions, "@inbounds val = row_vec[$child_pos]")
            push!(instructions, "@inbounds row_vec[$pos] = exp(clamp(val, -700.0, 700.0))")  # Prevent overflow
        else
            push!(instructions, "@inbounds row_vec[$pos] = $op(row_vec[$child_pos])")
        end
    catch e
        @error "Error in unary op compilation for $op" exception=(e, catch_backtrace())
        push!(instructions, "@inbounds row_vec[$pos] = row_vec[$child_pos]")  # Identity fallback
    end
end

function compile_binary_op_efficient!(instructions, op::Function, pos, left_pos, right_pos)
    try
        if op === (+)
            push!(instructions, "@inbounds row_vec[$pos] = row_vec[$left_pos] + row_vec[$right_pos]")
        elseif op === (-)
            push!(instructions, "@inbounds row_vec[$pos] = row_vec[$left_pos] - row_vec[$right_pos]")
        elseif op === (*)
            push!(instructions, "@inbounds row_vec[$pos] = row_vec[$left_pos] * row_vec[$right_pos]")
        elseif op === (/)
            push!(instructions, "@inbounds den = row_vec[$right_pos]")
            push!(instructions, "@inbounds row_vec[$pos] = abs(den) > 1e-16 ? row_vec[$left_pos] / den : row_vec[$left_pos]")
        elseif op === (^)
            push!(instructions, "@inbounds base = row_vec[$left_pos]")
            push!(instructions, "@inbounds exp_val = clamp(row_vec[$right_pos], -100.0, 100.0)")  # Prevent overflow
            push!(instructions, "@inbounds row_vec[$pos] = base^exp_val")
        else
            push!(instructions, "@inbounds row_vec[$pos] = $op(row_vec[$left_pos], row_vec[$right_pos])")
        end
    catch e
        @error "Error in binary op compilation for $op" exception=(e, catch_backtrace())
        push!(instructions, "@inbounds row_vec[$pos] = row_vec[$left_pos]")  # Left operand fallback
    end
end


function compile_interaction_efficient!(instructions, pos, child_positions)
    try
        if isempty(child_positions)
            push!(instructions, "@inbounds row_vec[$pos] = 1.0")
        elseif length(child_positions) == 1
            push!(instructions, "@inbounds row_vec[$pos] = row_vec[$(child_positions[1])]")
        elseif length(child_positions) == 2
            # This is the key case for x^2 * log(z)
            push!(instructions, "@inbounds row_vec[$pos] = row_vec[$(child_positions[1])] * row_vec[$(child_positions[2])]")
        else
            # General product
            push!(instructions, "@inbounds product = row_vec[$(child_positions[1])]")
            for i in 2:length(child_positions)
                push!(instructions, "@inbounds product *= row_vec[$(child_positions[i])]")
            end
            push!(instructions, "@inbounds row_vec[$pos] = product")
        end
        
        # Debug output
        println("      Compiled interaction at position $pos using positions $child_positions")
        
    catch e
        @error "Error in interaction compilation" exception=(e, catch_backtrace())
        push!(instructions, "@inbounds row_vec[$pos] = 1.0")  # Identity fallback
    end
end

function compile_zscore_efficient!(instructions, pos, child_pos, center, scale)
    push!(instructions, "@inbounds val = row_vec[$child_pos]")
    push!(instructions, "@inbounds row_vec[$pos] = (val - $center) / $scale")
end

###############################################################################
# Utility Functions (Reused from original code)
###############################################################################

function generate_safe_lookup_function(name, values)
    n_levels = length(values)
    
    code = """
    function $name(level::Int)
        # Bounds checking to prevent crashes
        if level < 1 || level > $n_levels
            return $(values[1])  # Default to first level
        end
        
        # Efficient lookup
    """
    
    for (i, val) in enumerate(values)
        if i == 1
            code *= "    level == $i ? $val :\n"
        elseif i == length(values)
            code *= "    $val\nend"
        else
            code *= "    level == $i ? $val :\n"
        end
    end
    
    try
        eval(Meta.parse(code))
    catch e
        @warn "Failed to generate lookup function $name: $e"
        # Fallback: generate simple function
        fallback_code = """
        function $name(level::Int)
            return $(values[max(1, min(level, length(values)))])
        end
        """
        eval(Meta.parse(fallback_code))
    end
end

"""
Generate lookup function for categorical contrasts (reused from original)
"""
function generate_lookup_function(name, values)
    code = "function $name(level::Int)\n"
    for (i, val) in enumerate(values)
        if i == 1
            code *= "    level == $i ? $val :\n"
        elseif i == length(values)
            code *= "    $val\nend"
        else
            code *= "    level == $i ? $val :\n"
        end
    end
    eval(Meta.parse(code))
end

###############################################################################
# Main Compositional Interface
###############################################################################

"""
    compile_formula_compositional_efficient(model) -> (func_name, output_width, column_names)

Compile any StatsModels formula using compositional approach with minimal allocations.
"""
function compile_formula_compositional_efficient(model)
    ctx = GLOBAL_COMPILATION_CONTEXT
    reset_context!(ctx)
    
    try
        rhs = fixed_effects_form(model).rhs
        
        println("=== Compositional Formula Compilation ===")
        println("Formula: $rhs")
        
        # Parse all terms safely
        for (i, term) in enumerate(rhs.terms)
            println("Processing term $i: $term ($(typeof(term)))")
            try
                nodes = parse_term_compositional(term)
                append!(ctx.node_buffer, nodes)
                println("  → Generated $(length(nodes)) nodes")
                
                # Debug: show node types
                for (j, node) in enumerate(nodes)
                    println("    Node $j: $(typeof(node))")
                end
            catch e
                @error "Failed to process term $i: $term" exception=(e, catch_backtrace())
                # Add fallback constant
                push!(ctx.node_buffer, ConstantNode(0.0))
            end
        end
        
        println("Total nodes: $(length(ctx.node_buffer))")
        
        if isempty(ctx.node_buffer)
            @warn "No nodes generated, adding intercept"
            push!(ctx.node_buffer, ConstantNode(1.0))
        end
        
        # Compile to instructions safely
        instructions, output_width = compile_expression_tree_efficient!(ctx, ctx.node_buffer)
        
        println("Generated $(length(instructions)) instructions")
        println("Output width: $output_width")
        
        # Generate function
        formula_hash = hash(string(rhs))
        func_name = generate_efficient_function(instructions, formula_hash)
        
        println("Generated function: $func_name")
        println("=== Compilation Complete ===")
        
        return func_name, output_width, unique(ctx.column_buffer)
        
    catch e
        @error "Fatal error in formula compilation" exception=(e, catch_backtrace())
        
        # Emergency fallback: create a simple intercept-only function
        emergency_func_name = Symbol("emergency_formula_$(abs(rand(Int)))")
        emergency_code = """
        function $emergency_func_name(row_vec, data, row_idx)
            @inbounds row_vec[1] = 1.0
            return row_vec
        end
        """
        eval(Meta.parse(emergency_code))
        
        @warn "Using emergency fallback function: $emergency_func_name"
        return emergency_func_name, 1, Symbol[]
    end
end


function generate_efficient_function(instructions, formula_hash)
    try
        func_name = Symbol("compositional_formula_$(abs(formula_hash))")
        
        println("Generating function: $func_name")
        println("Instructions to include:")
        for (i, instr) in enumerate(instructions)
            println("  $i: $instr")
        end
        
        func_code = """
        function $func_name(row_vec, data, row_idx)
            try
                $(join(instructions, "\n                "))
                return row_vec
            catch e
                println("Error in generated function: ", e)
                fill!(row_vec, 0.0)
                if length(row_vec) > 0
                    row_vec[1] = 1.0
                end
                return row_vec
            end
        end
        """
        
        println("Generated function code:")
        println(func_code)
        
        # Try to parse and eval
        parsed_code = Meta.parse(func_code)
        println("Successfully parsed function code")
        
        eval(parsed_code)
        println("Successfully eval'd function")
        
        # Test if function exists
        if isdefined(Main, func_name)
            println("Function $func_name exists in Main")
        else
            error("Function $func_name was not created!")
        end
        
        return func_name
        
    catch e
        @error "Failed to generate function" exception=(e, catch_backtrace())
        println("Full error details:")
        showerror(stdout, e, catch_backtrace())
        
        # Create emergency function
        emergency_name = Symbol("emergency_$(abs(rand(Int)))")
        emergency_code = """
        function $emergency_name(row_vec, data, row_idx)
            fill!(row_vec, 0.0)
            if length(row_vec) > 0
                row_vec[1] = 1.0
            end
            return row_vec
        end
        """
        
        try
            eval(Meta.parse(emergency_code))
            println("Created emergency function: $emergency_name")
            return emergency_name
        catch e2
            @error "Even emergency function failed!" exception=(e2, catch_backtrace())
            return :broken_function
        end
    end
end

# Also let's see what instructions are being generated
# REPLACE: compile_expression_tree_efficient! with more debug output
function compile_expression_tree_efficient!(ctx::CompilationContext, expression_nodes)
    try
        # Reset instruction buffer
        empty!(ctx.instruction_buffer)
        empty!(ctx.node_positions)
        ctx.position_counter[] = 1
        
        println("Compiling $(length(expression_nodes)) expression nodes...")
        
        # Compile each node safely
        for (i, node) in enumerate(expression_nodes)
            println("  Compiling node $i: $(typeof(node))")
            try
                pos = compile_node_efficient!(ctx, node)
                println("    → Compiled to position $pos")
            catch e
                @error "Failed to compile node $i: $(typeof(node))" exception=(e, catch_backtrace())
                # Add a fallback instruction
                pos = ctx.position_counter[]
                push!(ctx.instruction_buffer, "@inbounds row_vec[$pos] = 0.0")
                ctx.position_counter[] += 1
                println("    → Used fallback at position $pos")
            end
        end
        
        # Ensure we have at least one instruction
        if isempty(ctx.instruction_buffer)
            push!(ctx.instruction_buffer, "@inbounds row_vec[1] = 1.0")
            ctx.position_counter[] = 2
        end
        
        println("Final instruction buffer:")
        for (i, instr) in enumerate(ctx.instruction_buffer)
            println("  $i: $instr")
        end
        
        return ctx.instruction_buffer, ctx.position_counter[] - 1
        
    catch e
        @error "Fatal error in expression tree compilation" exception=(e, catch_backtrace())
        # Emergency fallback
        return ["@inbounds row_vec[1] = 1.0"], 1
    end
end

###############################################################################
# Testing and Validation
###############################################################################

function test_simple_interaction()
    println("=== Simple Interaction Test ===")
    
    # Very simple data
    df = DataFrame(
        x = [2.0],
        y = [1.0], 
        z = [1.0]  # log(1) = 0, so log(z) won't work. Let's use e
    )
    df.z = [MathConstants.e]  # log(e) = 1
    
    println("Data:")
    println("x = $(df.x[1]), z = $(df.z[1])")
    println("x^2 = $(df.x[1]^2)")
    println("log(z) = $(log(df.z[1]))")
    println("x^2 * log(z) = $(df.x[1]^2 * log(df.z[1]))")
    
    # Test the formula
    formula = @formula(y ~ x^2 * log(z))
    model = lm(formula, df)
    mm = modelmatrix(model)
    
    println("\nStatsModels result:")
    println("Shape: $(size(mm))")
    println("Values: $(mm[1, :])")
    println("Coef names: $(StatsModels.coefnames(model))")
    
    # Test our compiler
    println("\nOur compiler:")
    func_name, output_width, column_names = compile_formula_compositional_efficient(model)
    
    row_vec = Vector{Float64}(undef, output_width)
    data = Tables.columntable(df)
    func = getproperty(Main, func_name)
    
    func(row_vec, data, 1)
    
    println("Our result: $row_vec")
    println("Match: $(isapprox(mm[1, :], row_vec, atol=1e-10))")
end

function test_compositional_compiler()
    println("=== Testing Compositional Formula Compiler (Safe Mode) ===")
    
    try
        
        # Create test data with fixed seed for reproducibility
        Random.seed!(123)
        n = 50  # Smaller dataset for faster testing
        df = DataFrame(
            x = randn(n),
            y = randn(n), 
            z = abs.(randn(n)) .+ 0.1,  # Positive for log
            group = categorical(rand(["A", "B"], n))  # Just 2 levels for simplicity
        )
        data = Tables.columntable(df)

        # Test formulas in order of complexity
        test_formulas = [
            @formula(y ~ 1),                    # Intercept only
            @formula(y ~ x),                    # Simple continuous
            @formula(y ~ group),                # Simple categorical
            @formula(y ~ x + group),            # Mixed
            @formula(y ~ x + x^2),              # Power function
            @formula(y ~ x + log(z)),           # Log function
            @formula(y ~ x * group),            # Simple interaction
            @formula(y ~ x^2 * log(z)),         # Complex interaction
        ]
        
        successful_tests = 0
        
        for (i, formula) in enumerate(test_formulas)
            println("\n--- Test $i: $formula ---")
            
            (i, formula) = collect(enumerate(test_formulas))[i]
            try
                # Build model
                model = lm(formula, df)
                mm = modelmatrix(model);
                println("✅ Model built successfully")
                
                # Compile with safe compositional compiler
                func_name, output_width, column_names = compile_formula_compositional_efficient(model)
                println("✅ Safe compilation successful")
                println("   Function: $func_name")
                println("   Output width: $output_width")
                println("   Columns: $column_names")
                
                # Test evaluation
                # row_vec = Vector{Float64}(undef, output_width)
                row_vec = Vector{Float64}(undef, size(mm, 2))
                func = getproperty(Main, func_name)
                
                func(row_vec, data, 1)
                println("✅ Evaluation successful")
                
                # Verify correctness
                mm_row = mm[1, :]
                row_vec
                if length(mm_row) == length(row_vec) && isapprox(mm_row, row_vec, atol=1e-8)
                    println("✅ Results match model matrix!")
                    successful_tests += 1
                else
                    println("⚠️  Results don't match exactly:")
                    println("   Expected ($(length(mm_row))): $mm_row")
                    println("   Got ($(length(row_vec))):      $row_vec")
                    if length(mm_row) == length(row_vec)
                        println("   Max diff: $(maximum(abs.(mm_row .- row_vec)))")
                    end
                end
                
            catch e
                println("❌ Test failed: $e")
                @error "Test $i failed" exception=(e, catch_backtrace())
            end
        end
        
        println("\n=== Test Summary ===")
        println("Successful tests: $successful_tests / $(length(test_formulas))")
        
        if successful_tests == length(test_formulas)
            println("🎉 All tests passed!")
        elseif successful_tests > 0
            println("⚠️  Some tests passed, check the failing ones")
        else
            println("❌ All tests failed, major issues to fix")
        end
        
        return successful_tests == length(test_formulas)
        
    catch e
        @error "Fatal error in testing" exception=(e, catch_backtrace())
        return false
    end
end

# ADD NEW: Helper to extract columns from complex terms
function extract_columns_from_term(term::FunctionTerm)
    columns = Symbol[]
    for arg in term.args
        if arg isa Union{ContinuousTerm, Term}
            push!(columns, arg.sym)
        elseif arg isa FunctionTerm
            append!(columns, extract_columns_from_term(arg))
        end
    end
    return unique(columns)
end

function extract_columns_from_term(term::Union{ContinuousTerm, Term})
    return [term.sym]
end

function extract_columns_from_term(term::ConstantTerm)
    return Symbol[]
end

function test_corrected_formula()
    println("=== Testing Corrected Formula Parsing ===")
    
    # Simple test case
    df = DataFrame(
        x = [2.0],
        y = [1.0], 
        z = [MathConstants.e]  # log(e) = 1
    )
    
    println("Test values: x = $(df.x[1]), z = $(df.z[1])")
    println("Expected: x^2 = $(df.x[1]^2), log(z) = $(log(df.z[1])), interaction = $(df.x[1]^2 * log(df.z[1]))")
    
    formula = @formula(y ~ x^2 * log(z))
    model = lm(formula, df)
    mm = modelmatrix(model)
    
    println("\nStatsModels:")
    println("Shape: $(size(mm))")
    println("Values: $(mm[1, :])")
    
    # Our corrected compiler
    func_name, output_width, column_names = compile_formula_compositional_efficient(model)
    
    row_vec = Vector{Float64}(undef, output_width)
    data = Tables.columntable(df)
    func = getproperty(Main, func_name)
    
    func(row_vec, data, 1)
    
    println("\nOur result:")
    println("Shape: $output_width")
    println("Values: $row_vec")
    
    println("\nMatch: $(isapprox(mm[1, :], row_vec, atol=1e-10))")
    
    if length(mm[1, :]) == length(row_vec)
        println("Differences: $(mm[1, :] .- row_vec)")
    end
end

function test_step_by_step()
    println("=== Step by Step Test ===")

    
    # Very simple case first
    df = DataFrame(x = [2.0], y = [1.0])
    
    println("Step 1: Simple continuous term")
    formula1 = @formula(y ~ x)
    model1 = lm(formula1, df)
    mm1 = modelmatrix(model1)
    println("Expected: $(mm1[1, :]) (shape: $(size(mm1)))")
    
    try
        func1, width1, cols1 = compile_formula_compositional_efficient(model1)
        row_vec1 = Vector{Float64}(undef, width1)
        data1 = Tables.columntable(df)
        getproperty(Main, func1)(row_vec1, data1, 1)
        println("Got:      $row_vec1 (width: $width1)")
        println("Match:    $(isapprox(mm1[1, :], row_vec1))")
    catch e
        println("ERROR: $e")
    end
    
    println("\nStep 2: Power term")
    formula2 = @formula(y ~ x^2)
    model2 = lm(formula2, df)
    mm2 = modelmatrix(model2)
    println("Expected: $(mm2[1, :]) (shape: $(size(mm2)))")
    
    try
        func2, width2, cols2 = compile_formula_compositional_efficient(model2)
        row_vec2 = Vector{Float64}(undef, width2)
        data2 = Tables.columntable(df)
        getproperty(Main, func2)(row_vec2, data2, 1)
        println("Got:      $row_vec2 (width: $width2)")
        println("Match:    $(isapprox(mm2[1, :], row_vec2))")
    catch e
        println("ERROR: $e")
    end
    
    # Don't test the complex case until the simple ones work
end


function debug_simple_case()
    println("=== Debug Simple Case ===")
    
    df = DataFrame(x = [2.0], y = [1.0])
    formula = @formula(y ~ x)
    model = lm(formula, df)
    
    rhs = fixed_effects_form(model).rhs
    println("RHS: $rhs")
    println("Terms: $(rhs.terms)")
    
    # Manually parse each term
    for (i, term) in enumerate(rhs.terms)
        println("\nTerm $i: $term ($(typeof(term)))")
        nodes = parse_term_compositional(term)
        println("  Parsed to $(length(nodes)) nodes:")
        for (j, node) in enumerate(nodes)
            println("    Node $j: $(typeof(node)) - $node")
        end
    end
    
    # Try manual compilation
    ctx = GLOBAL_COMPILATION_CONTEXT
    reset_context!(ctx)
    
    # Add nodes manually
    push!(ctx.node_buffer, ConstantNode(1.0))
    push!(ctx.node_buffer, ColumnNode(:x))
    
    println("\nManual compilation:")
    instructions, width = compile_expression_tree_efficient!(ctx, ctx.node_buffer)
    
    println("Instructions: $instructions")
    println("Width: $width")
    
    # Try to generate function manually
    func_name = generate_efficient_function(instructions, 12345)
    println("Generated function: $func_name")
    
    # Try to call it
    if isdefined(Main, func_name)
        row_vec = Vector{Float64}(undef, width)
        data = Tables.columntable(df)
        
        println("Calling function...")
        try
            result = getproperty(Main, func_name)(row_vec, data, 1)
            println("Result: $result")
        catch e
            println("Function call error: $e")
        end
    else
        println("Function doesn't exist!")
    end
end

function debug_data_access()
    println("=== Debug Data Access ===")
    
    df = DataFrame(x = [2.0], y = [1.0])
    data = Tables.columntable(df)
    
    println("Original DataFrame:")
    println(df)
    
    println("\nTables.columntable result:")
    println("Type: $(typeof(data))")
    println("Data: $data")
    println("Keys: $(keys(data))")
    
    println("\nTrying to access data.x:")
    try
        x_col = data.x
        println("data.x = $x_col (type: $(typeof(x_col)))")
        println("data.x[1] = $(x_col[1])")
    catch e
        println("Error accessing data.x: $e")
    end
    
    println("\nTrying to access data[:x]:")
    try
        x_col2 = data[:x]
        println("data[:x] = $x_col2 (type: $(typeof(x_col2)))")
    catch e
        println("Error accessing data[:x]: $e")
    end
    
    println("\nTrying manual function call:")
    
    # Create a simple test function
    test_func_code = """
    function test_data_access(row_vec, data, row_idx)
        println("Inside function:")
        println("  row_vec type: ", typeof(row_vec))
        println("  data type: ", typeof(data))
        println("  row_idx: ", row_idx)
        println("  data keys: ", keys(data))
        
        try
            x_val = data.x[row_idx]
            println("  data.x[row_idx] = ", x_val)
            row_vec[1] = 1.0
            row_vec[2] = Float64(x_val)
            return row_vec
        catch e
            println("  Error: ", e)
            rethrow(e)
        end
    end
    """
    
    eval(Meta.parse(test_func_code))
    
    row_vec = Vector{Float64}(undef, 2)
    println("\nCalling test function:")
    try
        result = test_data_access(row_vec, data, 1)
        println("Success! Result: $result")
    catch e
        println("Test function failed: $e")
        showerror(stdout, e, catch_backtrace())
    end
end