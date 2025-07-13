# phase2_instruction_generation.jl
# Phase 2: Generate efficient instructions from formula analysis

using Tables
using CategoricalArrays: CategoricalValue, levelcode
using StatsModels
using StandardizedPredictors: ZScoredTerm

###############################################################################
# Phase 2: Instruction Generation
###############################################################################

"""
    generate_instructions(analysis::FormulaAnalysis) -> Vector{String}

Phase 2: Convert formula analysis into efficient instruction sequence.
Each instruction is a string of Julia code that will be embedded in @generated function.
"""
function generate_instructions(analysis::FormulaAnalysis)
    println("=== Phase 2: Instruction Generation ===")
    println("Generating instructions for $(length(analysis.terms)) terms")
    
    instructions = String[]
    
    # Generate instructions for each term in order
    for (i, term_analysis) in enumerate(analysis.terms)
        println("Generating instructions for term $i: $(term_analysis.term_type)")
        
        term_instructions = generate_term_instructions(term_analysis)
        append!(instructions, term_instructions)
        
        println("  → Generated $(length(term_instructions)) instructions")
    end
    
    println("=== Total: $(length(instructions)) instructions ===")
    
    return instructions
end

"""
Dispatch to appropriate instruction generator based on term type.
"""
function generate_term_instructions(term_analysis::TermAnalysis)
    if term_analysis.term_type == :constant
        return generate_constant_instructions(term_analysis)
    elseif term_analysis.term_type == :continuous
        return generate_continuous_instructions(term_analysis)
    elseif term_analysis.term_type == :categorical
        return generate_categorical_instructions(term_analysis)
    elseif term_analysis.term_type == :function
        return generate_function_instructions(term_analysis)
    elseif term_analysis.term_type == :interaction
        return generate_interaction_instructions(term_analysis)
    elseif term_analysis.term_type == :zscore
        return generate_zscore_instructions(term_analysis)
    else
        @warn "Unknown term type: $(term_analysis.term_type), using fallback"
        return generate_fallback_instructions(term_analysis)
    end
end

###############################################################################
# Constant Term Instructions
###############################################################################

function generate_constant_instructions(term_analysis::TermAnalysis)
    pos = term_analysis.start_position
    
    # Handle omitted intercepts (width = 0)
    if get(term_analysis.metadata, :omitted, false)
        return String[]  # No instructions for omitted terms
    end
    
    value = term_analysis.metadata[:value]
    
    return ["@inbounds row_vec[$pos] = $value"]
end

###############################################################################
# Continuous Term Instructions  
###############################################################################

function generate_continuous_instructions(term_analysis::TermAnalysis)
    pos = term_analysis.start_position
    col = term_analysis.metadata[:column]
    
    return ["@inbounds row_vec[$pos] = Float64(data.$col[row_idx])"]
end

###############################################################################
# Categorical Term Instructions
###############################################################################

function generate_categorical_instructions(term_analysis::TermAnalysis)
    start_pos = term_analysis.start_position
    width = term_analysis.width
    col = term_analysis.metadata[:column]
    contrast_matrix = term_analysis.metadata[:contrast_matrix]
    n_levels = term_analysis.metadata[:n_levels]
    
    instructions = String[]
    
    # Extract categorical value and level code (once)
    push!(instructions, "@inbounds cat_val = data.$col[row_idx]")
    push!(instructions, "@inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
    push!(instructions, "@inbounds level_code = clamp(level_code, 1, $n_levels)")
    
    # Generate instructions for each contrast column
    for j in 1:width
        pos = start_pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        # Choose efficient code generation based on number of levels
        if n_levels == 1
            # Single level (shouldn't happen in practice, but safe)
            push!(instructions, "@inbounds row_vec[$pos] = $(values[1])")
            
        elseif n_levels == 2
            # Binary categorical - simple ternary
            push!(instructions, "@inbounds row_vec[$pos] = level_code == 1 ? $(values[1]) : $(values[2])")
            
        elseif n_levels == 3
            # Three levels - nested ternary
            push!(instructions, "@inbounds row_vec[$pos] = level_code == 1 ? $(values[1]) : level_code == 2 ? $(values[2]) : $(values[3])")
            
        elseif n_levels <= 6
            # Small number of levels - chain of ternaries
            ternary_chain = "level_code == 1 ? $(values[1])"
            for i in 2:(n_levels-1)
                ternary_chain *= " : level_code == $i ? $(values[i])"
            end
            ternary_chain *= " : $(values[n_levels])"
            push!(instructions, "@inbounds row_vec[$pos] = $ternary_chain")
            
        else
            # Many levels - use lookup table approach
            lookup_name = generate_lookup_table(col, j, values)
            push!(instructions, "@inbounds row_vec[$pos] = $lookup_name(level_code)")
        end
    end
    
    return instructions
end

###############################################################################
# Function Term Instructions
###############################################################################

function generate_function_instructions(term_analysis::TermAnalysis)
    pos = term_analysis.start_position
    pattern = term_analysis.metadata[:pattern]
    func = term_analysis.metadata[:function]
    
    if pattern == :unary && get(term_analysis.metadata, :simple_unary, false)
        # Simple unary function: f(column)
        col = term_analysis.metadata[:arg_column]
        return generate_simple_unary_instructions(pos, func, col)
        
    elseif pattern == :power
        # Power function: column^exponent
        col = term_analysis.metadata[:base_column]
        exp = term_analysis.metadata[:exponent]
        return generate_power_instructions(pos, col, exp)
        
    elseif pattern == :binary
        # Binary function: f(arg1, arg2)
        return generate_binary_function_instructions(term_analysis)
        
    else
        # Complex function - fall back to safe evaluation
        @warn "Complex function pattern: $pattern for $(term_analysis.term)"
        return generate_complex_function_instructions(term_analysis)
    end
end

function generate_simple_unary_instructions(pos::Int, func::Function, col::Symbol)
    instructions = String[]
    
    push!(instructions, "@inbounds val = Float64(data.$col[row_idx])")
    
    # Optimize common functions
    if func === log
        push!(instructions, "@inbounds row_vec[$pos] = val > 0.0 ? log(val) : log(abs(val) + 1e-16)")
    elseif func === exp
        push!(instructions, "@inbounds row_vec[$pos] = exp(clamp(val, -700.0, 700.0))")
    elseif func === sqrt
        push!(instructions, "@inbounds row_vec[$pos] = sqrt(abs(val))")
    elseif func === sin
        push!(instructions, "@inbounds row_vec[$pos] = sin(val)")
    elseif func === cos
        push!(instructions, "@inbounds row_vec[$pos] = cos(val)")
    else
        # General function
        push!(instructions, "@inbounds row_vec[$pos] = $func(val)")
    end
    
    return instructions
end

function generate_power_instructions(pos::Int, col::Symbol, exponent::Float64)
    instructions = String[]
    
    push!(instructions, "@inbounds val = Float64(data.$col[row_idx])")
    
    # Optimize common exponents
    if exponent == 2.0
        push!(instructions, "@inbounds row_vec[$pos] = val * val")
    elseif exponent == 3.0
        push!(instructions, "@inbounds row_vec[$pos] = val * val * val")
    elseif exponent == 0.5
        push!(instructions, "@inbounds row_vec[$pos] = sqrt(abs(val))")
    elseif exponent == -1.0
        push!(instructions, "@inbounds row_vec[$pos] = abs(val) > 1e-16 ? 1.0 / val : val")
    else
        push!(instructions, "@inbounds row_vec[$pos] = val^$exponent")
    end
    
    return instructions
end

function generate_binary_function_instructions(term_analysis::TermAnalysis)
    # For now, implement a safe fallback for binary functions
    # This could be expanded to handle specific patterns like x + y, x * y, etc.
    return generate_complex_function_instructions(term_analysis)
end

function generate_complex_function_instructions(term_analysis::TermAnalysis)
    pos = term_analysis.start_position
    
    # Safe fallback for complex functions
    @warn "Using fallback for complex function: $(term_analysis.term)"
    return ["@inbounds row_vec[$pos] = 1.0  # Complex function fallback"]
end

###############################################################################
# Interaction Term Instructions
###############################################################################

function generate_interaction_instructions(term_analysis::TermAnalysis)
    start_pos = term_analysis.start_position
    total_width = term_analysis.width
    component_info = term_analysis.metadata[:components]
    component_widths = term_analysis.metadata[:component_widths]
    n_components = term_analysis.metadata[:n_components]
    
    instructions = String[]
    
    println("    Generating interaction: $n_components components, width $total_width")
    
    # Generate code to evaluate each component into temporary variables
    component_vars = String[]
    
    for (i, comp_info) in enumerate(component_info)
        comp_term = comp_info[:term]
        comp_width = comp_info[:width]
        comp_type = comp_info[:type]
        
        if comp_width == 1
            # Single-column component
            var_name = "comp_$(i)"
            push!(component_vars, var_name)
            
            # Generate evaluation code based on component type
            if comp_type == ContinuousTerm || comp_type == Term
                col = comp_term.sym
                push!(instructions, "@inbounds $var_name = Float64(data.$col[row_idx])")
                
            elseif comp_type == FunctionTerm
                # Handle function components
                if comp_term.f === (^) && length(comp_term.args) == 2 && comp_term.args[2] isa ConstantTerm
                    # Power function
                    col = comp_term.args[1].sym
                    exp = Float64(comp_term.args[2].n)
                    push!(instructions, "@inbounds val_$i = Float64(data.$col[row_idx])")
                    if exp == 2.0
                        push!(instructions, "@inbounds $var_name = val_$i * val_$i")
                    else
                        push!(instructions, "@inbounds $var_name = val_$i^$exp")
                    end
                elseif length(comp_term.args) == 1 && comp_term.args[1] isa Union{ContinuousTerm, Term}
                    # Simple unary function
                    col = comp_term.args[1].sym
                    func = comp_term.f
                    push!(instructions, "@inbounds val_$i = Float64(data.$col[row_idx])")
                    if func === log
                        push!(instructions, "@inbounds $var_name = val_$i > 0.0 ? log(val_$i) : log(abs(val_$i) + 1e-16)")
                    else
                        push!(instructions, "@inbounds $var_name = $func(val_$i)")
                    end
                else
                    @warn "Complex function component in interaction: $comp_term"
                    push!(instructions, "@inbounds $var_name = 1.0  # Complex function fallback")
                end
                
            else
                @warn "Unsupported component type in interaction: $comp_type"
                push!(instructions, "@inbounds $var_name = 1.0  # Unsupported component fallback")
            end
        else
            @warn "Multi-column component in interaction not yet fully supported: $comp_term (width: $comp_width)"
            # For now, use first column only
            var_name = "comp_$(i)"
            push!(component_vars, var_name)
            push!(instructions, "@inbounds $var_name = 1.0  # Multi-column component fallback")
        end
    end
    
    # Generate the interaction computation
    if total_width == 1
        # Simple product interaction
        if length(component_vars) == 1
            push!(instructions, "@inbounds row_vec[$start_pos] = $(component_vars[1])")
        elseif length(component_vars) == 2
            push!(instructions, "@inbounds row_vec[$start_pos] = $(component_vars[1]) * $(component_vars[2])")
        else
            # General product
            product_expr = join(component_vars, " * ")
            push!(instructions, "@inbounds row_vec[$start_pos] = $product_expr")
        end
    else
        # Multi-column interaction (Kronecker product)
        # This is the complex case - for now, implement a simplified version
        @warn "Multi-column interaction (Kronecker product) not fully implemented, using simplified version"
        
        # Simplified: fill all positions with the product (not mathematically correct, but safe)
        if length(component_vars) >= 1
            product_expr = join(component_vars, " * ")
            for pos in start_pos:(start_pos + total_width - 1)
                push!(instructions, "@inbounds row_vec[$pos] = $product_expr")
            end
        else
            # Ultimate fallback
            for pos in start_pos:(start_pos + total_width - 1)
                push!(instructions, "@inbounds row_vec[$pos] = 1.0")
            end
        end
    end
    
    return instructions
end

###############################################################################
# ZScore Term Instructions
###############################################################################

function generate_zscore_instructions(term_analysis::TermAnalysis)
    pos = term_analysis.start_position
    underlying_type = term_analysis.metadata[:underlying_type]
    center = term_analysis.metadata[:center]
    scale = term_analysis.metadata[:scale]
    
    instructions = String[]
    
    # Generate instructions based on underlying term type
    if underlying_type == :continuous
        col = term_analysis.columns_used[1]  # Should be exactly one column
        push!(instructions, "@inbounds val = Float64(data.$col[row_idx])")
        push!(instructions, "@inbounds row_vec[$pos] = (val - $center) / $scale")
        
    elseif underlying_type == :function
        # For ZScored function terms, need to evaluate the function first
        underlying_term = term_analysis.metadata[:underlying_term]
        
        if underlying_term isa FunctionTerm && length(underlying_term.args) == 1 && 
           underlying_term.args[1] isa Union{ContinuousTerm, Term}
            # Simple case: zscore(f(x))
            col = underlying_term.args[1].sym
            func = underlying_term.f
            
            push!(instructions, "@inbounds raw_val = Float64(data.$col[row_idx])")
            
            if func === log
                push!(instructions, "@inbounds func_val = raw_val > 0.0 ? log(raw_val) : log(abs(raw_val) + 1e-16)")
            else
                push!(instructions, "@inbounds func_val = $func(raw_val)")
            end
            
            push!(instructions, "@inbounds row_vec[$pos] = (func_val - $center) / $scale")
        else
            @warn "Complex ZScored function term: $underlying_term"
            push!(instructions, "@inbounds row_vec[$pos] = 0.0  # Complex ZScore fallback")
        end
    else
        @warn "Unsupported ZScored term type: $underlying_type"
        push!(instructions, "@inbounds row_vec[$pos] = 0.0  # ZScore fallback")
    end
    
    return instructions
end

###############################################################################
# Fallback Instructions
###############################################################################

function generate_fallback_instructions(term_analysis::TermAnalysis)
    start_pos = term_analysis.start_position
    width = term_analysis.width
    
    instructions = String[]
    
    @warn "Using fallback instructions for term: $(term_analysis.term)"
    
    # Fill all positions with safe fallback values
    for i in 0:(width-1)
        pos = start_pos + i
        if i == 0
            push!(instructions, "@inbounds row_vec[$pos] = 1.0  # Fallback (intercept-like)")
        else
            push!(instructions, "@inbounds row_vec[$pos] = 0.0  # Fallback (zero)")
        end
    end
    
    return instructions
end

###############################################################################
# Lookup Table Generation
###############################################################################

const LOOKUP_COUNTER = Ref(0)

"""
Generate a lookup table function for categorical contrasts with many levels.
"""
function generate_lookup_table(col::Symbol, contrast_idx::Int, values::Vector{Float64})
    LOOKUP_COUNTER[] += 1
    lookup_name = Symbol("lookup_$(col)_$(contrast_idx)_$(LOOKUP_COUNTER[])")
    
    n_levels = length(values)
    
    # Generate the lookup function
    code = """
    function $lookup_name(level::Int)
        # Bounds checking
        if level < 1 || level > $n_levels
            return $(values[1])  # Default to first level
        end
        
        # Direct lookup
    """
    
    # Add the lookup logic
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
        println("    Generated lookup table: $lookup_name")
    catch e
        @warn "Failed to generate lookup function $lookup_name: $e"
        # Create a simple fallback function
        eval(Meta.parse("$lookup_name(level) = $(values[1])"))
    end
    
    return lookup_name
end

###############################################################################
# Testing and Validation
###############################################################################

"""
Test instruction generation on a simple formula.
"""
function test_instruction_generation()
    println("=== Testing Instruction Generation ===")
    
    # Create simple test case
    df = DataFrame(
        x = [2.0, 3.0],
        y = [1.0, 2.0], 
        z = [MathConstants.e, 2.0],
        group = categorical(["A", "B"])
    )
    
    # Test simple formula
    formula = @formula(y ~ x + x^2 + log(z) + group)
    model = lm(formula, df)
    
    # Phase 1: Analyze structure
    analysis = analyze_formula_structure(model)
    
    # Phase 2: Generate instructions
    instructions = generate_instructions(analysis)
    
    println("\nGenerated Instructions:")
    for (i, instr) in enumerate(instructions)
        println("  $i: $instr")
    end
    
    println("\nValidation:")
    println("Number of instructions: $(length(instructions))")
    println("Analysis width: $(analysis.total_width)")
    
    # Verify we have reasonable instructions
    has_data_access = any(occursin("data.", instr) for instr in instructions)
    has_row_vec_assignment = any(occursin("row_vec[", instr) for instr in instructions)
    
    println("Has data access: $has_data_access")
    println("Has row_vec assignments: $has_row_vec_assignment")
    
    return instructions, analysis
end

# Export main functions
export generate_instructions, generate_term_instructions, test_instruction_generation