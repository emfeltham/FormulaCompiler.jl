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

function generate_simple_product_interaction(component_info, start_pos, instructions)
    # All components are single-column, so just multiply them
    component_exprs = String[]
    
    for (i, comp_info) in enumerate(component_info)
        comp_term = comp_info[:term]
        comp_type = comp_info[:type]
        
        if comp_type <: ContinuousTerm || comp_type <: Term
            col = comp_term.sym
            push!(component_exprs, "Float64(data.$col[row_idx])")
            
        elseif comp_type <: CategoricalTerm
            # FIXED: Handle single-column categorical terms
            col = comp_term.sym
            contrast_matrix = comp_term.contrasts.matrix
            n_levels = size(contrast_matrix, 1)
            
            # Since this is width=1, we only have one contrast column (index 1)
            values = [contrast_matrix[level, 1] for level in 1:n_levels]
            
            # Generate inline categorical evaluation
            if n_levels == 2
                cat_expr = "(let cat_val = data.$col[row_idx], level_code = (cat_val isa CategoricalValue ? levelcode(cat_val) : 1); level_code == 1 ? $(values[1]) : $(values[2]) end)"
            elseif n_levels == 3
                cat_expr = "(let cat_val = data.$col[row_idx], level_code = (cat_val isa CategoricalValue ? levelcode(cat_val) : 1); level_code == 1 ? $(values[1]) : level_code == 2 ? $(values[2]) : $(values[3]) end)"
            else
                # For more levels, create ternary chain
                ternary_chain = "level_code == 1 ? $(values[1])"
                for level in 2:(n_levels-1)
                    ternary_chain *= " : level_code == $level ? $(values[level])"
                end
                ternary_chain *= " : $(values[n_levels])"
                cat_expr = "(let cat_val = data.$col[row_idx], level_code = clamp((cat_val isa CategoricalValue ? levelcode(cat_val) : 1), 1, $n_levels); $ternary_chain end)"
            end
            
            push!(component_exprs, cat_expr)
            
        elseif comp_type <: FunctionTerm
            # Handle function components inline (unchanged)
            if comp_term.f === (^) && length(comp_term.args) == 2 && comp_term.args[2] isa ConstantTerm
                # Power function: x^2
                col = comp_term.args[1].sym
                exp = Float64(comp_term.args[2].n)
                if exp == 2.0
                    push!(component_exprs, "(Float64(data.$col[row_idx]) * Float64(data.$col[row_idx]))")
                else
                    push!(component_exprs, "Float64(data.$col[row_idx])^$exp")
                end
            elseif length(comp_term.args) == 1 && comp_term.args[1] isa Union{ContinuousTerm, Term}
                # Simple unary function: log(z)
                col = comp_term.args[1].sym
                func = comp_term.f
                if func === log
                    val_expr = "Float64(data.$col[row_idx])"
                    push!(component_exprs, "($val_expr > 0.0 ? log($val_expr) : log(abs($val_expr) + 1e-16))")
                else
                    push!(component_exprs, "$func(Float64(data.$col[row_idx]))")
                end
            else
                @warn "Complex function component: $comp_term"
                push!(component_exprs, "1.0")
            end
            
        else
            @warn "Unsupported simple component type: $comp_type"
            push!(component_exprs, "1.0")
        end
    end
    
    # Generate the multiplication
    product_expr = join(component_exprs, " * ")
    push!(instructions, "@inbounds row_vec[$start_pos] = $product_expr")
    
    return instructions
end

function generate_binary_function_instructions(term_analysis::TermAnalysis)
    pos = term_analysis.start_position
    func = term_analysis.metadata[:function]
    args = term_analysis.metadata[:args]
    
    instructions = String[]
    
    # Handle binary comparison operators - FIXED FOR BOOLEAN CONVERSION
    if func in [>, <, >=, <=, ==, !=] && length(args) == 2
        arg1, arg2 = args
        
        if arg1 isa Union{ContinuousTerm, Term} && arg2 isa ConstantTerm
            # Simple case: column > constant - FIXED BOOLEAN CONVERSION
            col = arg1.sym
            const_val = Float64(arg2.n)
            
            push!(instructions, "@inbounds val = Float64(data.$col[row_idx])")
            
            # CRITICAL FIX: Proper boolean to float conversion
            if func === (>)
                push!(instructions, "@inbounds row_vec[$pos] = val > $const_val ? 1.0 : 0.0")
            elseif func === (<)
                push!(instructions, "@inbounds row_vec[$pos] = val < $const_val ? 1.0 : 0.0")
            elseif func === (>=)
                push!(instructions, "@inbounds row_vec[$pos] = val >= $const_val ? 1.0 : 0.0")
            elseif func === (<=)
                push!(instructions, "@inbounds row_vec[$pos] = val <= $const_val ? 1.0 : 0.0")
            elseif func === (==)
                push!(instructions, "@inbounds row_vec[$pos] = val == $const_val ? 1.0 : 0.0")
            elseif func === (!=)
                push!(instructions, "@inbounds row_vec[$pos] = val != $const_val ? 1.0 : 0.0")
            end
            
            return instructions
            
        elseif arg1 isa Union{ContinuousTerm, Term} && arg2 isa Union{ContinuousTerm, Term}
            # Both arguments are variables
            col1, col2 = arg1.sym, arg2.sym
            
            push!(instructions, "@inbounds val1 = Float64(data.$col1[row_idx])")
            push!(instructions, "@inbounds val2 = Float64(data.$col2[row_idx])")
            
            if func === (>)
                push!(instructions, "@inbounds row_vec[$pos] = val1 > val2 ? 1.0 : 0.0")
            elseif func === (<)
                push!(instructions, "@inbounds row_vec[$pos] = val1 < val2 ? 1.0 : 0.0")
            elseif func === (>=)
                push!(instructions, "@inbounds row_vec[$pos] = val1 >= val2 ? 1.0 : 0.0")
            elseif func === (<=)
                push!(instructions, "@inbounds row_vec[$pos] = val1 <= val2 ? 1.0 : 0.0")
            elseif func === (==)
                push!(instructions, "@inbounds row_vec[$pos] = val1 == val2 ? 1.0 : 0.0")
            elseif func === (!=)
                push!(instructions, "@inbounds row_vec[$pos] = val1 != val2 ? 1.0 : 0.0")
            end
            
            return instructions
        end
    end
    
    # Handle other binary arithmetic operators (unchanged)
    if func in [+, -, *, /, ^] && length(args) == 2
        arg1, arg2 = args
        
        if arg1 isa Union{ContinuousTerm, Term} && arg2 isa ConstantTerm
            col = arg1.sym
            const_val = Float64(arg2.n)
            
            push!(instructions, "@inbounds val = Float64(data.$col[row_idx])")
            
            if func === (+)
                push!(instructions, "@inbounds row_vec[$pos] = val + $const_val")
            elseif func === (-)
                push!(instructions, "@inbounds row_vec[$pos] = val - $const_val")
            elseif func === (*)
                push!(instructions, "@inbounds row_vec[$pos] = val * $const_val")
            elseif func === (/)
                push!(instructions, "@inbounds row_vec[$pos] = abs($const_val) > 1e-16 ? val / $const_val : val")
            elseif func === (^)
                push!(instructions, "@inbounds row_vec[$pos] = val^$const_val")
            end
            
            return instructions
        end
    end
    
    # For other binary functions, fall back to complex function handling
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
    
    # Special case: All single-column components (simple product)
    if all(w == 1 for w in component_widths)
        return generate_simple_product_interaction(component_info, start_pos, instructions)
    end
    
    # Mixed case: Some multi-column components (proper Kronecker product)
    return generate_kronecker_interaction(component_info, component_widths, start_pos, total_width, instructions)
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
# Kronecker Instructions
###############################################################################

function generate_kronecker_interaction(component_info, component_widths, start_pos, total_width, instructions)
    # Handle different cases based on number of components
    n_components = length(component_info)
    
    if n_components == 2
        comp1_info, comp2_info = component_info
        w1, w2 = component_widths
        
        if w1 == 1 && w2 > 1
            # First component is scalar, second is vector
            return generate_scalar_vector_kronecker(comp1_info, comp2_info, w2, start_pos, instructions)
            
        elseif w1 > 1 && w2 == 1
            # First component is vector, second is scalar
            return generate_scalar_vector_kronecker(comp1_info, comp2_info, w1, start_pos, instructions)
            
        elseif w1 > 1 && w2 > 1
            # Both multi-column - full Kronecker product
            return generate_full_kronecker_product(comp1_info, comp2_info, w1, w2, start_pos, instructions)
        end
        
    elseif n_components == 3
        # Three-way interaction - IMPLEMENT FOR TEST 11
        return generate_three_way_interaction(component_info, component_widths, start_pos, total_width, instructions)
        
    else
        # More than 3 components - general fallback
        @warn "$(n_components)-way interaction not fully implemented, using fallback"
        for i in 0:(total_width-1)
            push!(instructions, "@inbounds row_vec[$(start_pos + i)] = 1.0  # General fallback")
        end
        return instructions
    end
    
    # Fallback for unhandled 2-component cases
    @warn "Unhandled 2-component Kronecker case, using fallback"
    for i in 0:(total_width-1)
        push!(instructions, "@inbounds row_vec[$(start_pos + i)] = 1.0  # 2-component fallback")
    end
    
    return instructions
end

function generate_scalar_vector_kronecker(comp1_info, comp2_info, w2, start_pos, instructions)
    # comp1 is scalar, comp2 is vector (e.g., x * group)
    comp1_term = comp1_info[:term]
    comp2_term = comp2_info[:term]
    
    # Generate scalar value
    if comp1_term isa Union{ContinuousTerm, Term}
        col1 = comp1_term.sym
        push!(instructions, "@inbounds scalar_val = Float64(data.$col1[row_idx])")
    else
        @warn "Complex scalar component in Kronecker: $comp1_term"
        push!(instructions, "@inbounds scalar_val = 1.0")
    end
    
    # Generate vector values and multiply - FIXED VARIABLE NAMES
    if comp2_term isa CategoricalTerm
        col2 = comp2_term.sym
        contrast_matrix = comp2_term.contrasts.matrix
        n_levels = size(contrast_matrix, 1)
        
        # Get categorical level (once)
        push!(instructions, "@inbounds cat_val = data.$col2[row_idx]")
        push!(instructions, "@inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
        push!(instructions, "@inbounds level_code = clamp(level_code, 1, $n_levels)")
        
        # CRITICAL FIX: Use unique variable names for each contrast column
        for j in 1:w2
            pos = start_pos + j - 1
            values = [contrast_matrix[i, j] for i in 1:n_levels]
            
            # Use unique variable name for each column
            if n_levels == 2
                push!(instructions, "@inbounds contrast_val_$j = level_code == 1 ? $(values[1]) : $(values[2])")
            elseif n_levels == 3
                push!(instructions, "@inbounds contrast_val_$j = level_code == 1 ? $(values[1]) : level_code == 2 ? $(values[2]) : $(values[3])")
            else
                ternary_chain = "level_code == 1 ? $(values[1])"
                for i in 2:(n_levels-1)
                    ternary_chain *= " : level_code == $i ? $(values[i])"
                end
                ternary_chain *= " : $(values[n_levels])"
                push!(instructions, "@inbounds contrast_val_$j = $ternary_chain")
            end
            
            # Multiply by scalar using the unique variable
            push!(instructions, "@inbounds row_vec[$pos] = scalar_val * contrast_val_$j")
        end
    else
        @warn "Complex vector component in Kronecker: $comp2_term"
        for j in 1:w2
            pos = start_pos + j - 1
            push!(instructions, "@inbounds row_vec[$pos] = scalar_val")
        end
    end
    
    return instructions
end

function generate_full_kronecker_product(comp1_info, comp2_info, w1, w2, start_pos, instructions)
    # Both components are multi-column (e.g., group1 * group2)
    # Implement proper categorical × categorical Kronecker product
    println("DEBUG: Entering generate_full_kronecker_product with w1=$w1, w2=$w2")

    comp1_term = comp1_info[:term]
    comp2_term = comp2_info[:term]
    
    # Both should be categorical terms
    println("DEBUG: comp1_term type: $(typeof(comp1_term))")
    println("DEBUG: comp2_term type: $(typeof(comp2_term))")
    if comp1_term isa CategoricalTerm && comp2_term isa CategoricalTerm
        col1 = comp1_term.sym
        col2 = comp2_term.sym
        contrast_matrix1 = comp1_term.contrasts.matrix
        contrast_matrix2 = comp2_term.contrasts.matrix
        n_levels1 = size(contrast_matrix1, 1)
        n_levels2 = size(contrast_matrix2, 1)
        
        # Get categorical levels (once)
        push!(instructions, "@inbounds cat_val1 = data.$col1[row_idx]")
        push!(instructions, "@inbounds level_code1 = cat_val1 isa CategoricalValue ? levelcode(cat_val1) : 1")
        push!(instructions, "@inbounds level_code1 = clamp(level_code1, 1, $n_levels1)")
        
        push!(instructions, "@inbounds cat_val2 = data.$col2[row_idx]")
        push!(instructions, "@inbounds level_code2 = cat_val2 isa CategoricalValue ? levelcode(cat_val2) : 1")
        push!(instructions, "@inbounds level_code2 = clamp(level_code2, 1, $n_levels2)")
        
        # Generate Kronecker product in StatsModels order: [c1*d1, c2*d1, c1*d2, c2*d2]
        col_idx = 0
        for j in 1:w2  # Second component columns  
            for i in 1:w1  # First component columns
                pos = start_pos + col_idx
                
                # Get contrast values for current levels
                values1 = [contrast_matrix1[level, i] for level in 1:n_levels1]
                values2 = [contrast_matrix2[level, j] for level in 1:n_levels2]
                
                # Generate efficient lookup for both contrasts
                if n_levels1 == 2
                    contrast1_expr = "level_code1 == 1 ? $(values1[1]) : $(values1[2])"
                elseif n_levels1 == 3
                    contrast1_expr = "level_code1 == 1 ? $(values1[1]) : level_code1 == 2 ? $(values1[2]) : $(values1[3])"
                else
                    ternary_chain = "level_code1 == 1 ? $(values1[1])"
                    for level in 2:(n_levels1-1)
                        ternary_chain *= " : level_code1 == $level ? $(values1[level])"
                    end
                    ternary_chain *= " : $(values1[n_levels1])"
                    contrast1_expr = ternary_chain
                end
                
                if n_levels2 == 2
                    contrast2_expr = "level_code2 == 1 ? $(values2[1]) : $(values2[2])"
                elseif n_levels2 == 3
                    contrast2_expr = "level_code2 == 1 ? $(values2[1]) : level_code2 == 2 ? $(values2[2]) : $(values2[3])"
                else
                    ternary_chain = "level_code2 == 1 ? $(values2[1])"
                    for level in 2:(n_levels2-1)
                        ternary_chain *= " : level_code2 == $level ? $(values2[level])"
                    end
                    ternary_chain *= " : $(values2[n_levels2])"
                    contrast2_expr = ternary_chain
                end
                
                # Compute Kronecker product element
                push!(instructions, "@inbounds row_vec[$pos] = ($contrast1_expr) * ($contrast2_expr)")
                
                col_idx += 1
            end
        end
        
    else
        # Fallback for non-categorical cases
        @warn "Complex full Kronecker product: $comp1_term × $comp2_term"
        total_width = w1 * w2
        for idx in 0:(total_width-1)
            pos = start_pos + idx
            push!(instructions, "@inbounds row_vec[$pos] = 1.0  # Non-categorical fallback")
        end
    end
    
    println("DEBUG: Generated instructions:")
    for (i, instr) in enumerate(instructions[(end-9):end])  # Show last 10 instructions
        println("  $i: $instr")
    end
    return instructions
end

function generate_three_way_interaction(component_info, component_widths, start_pos, total_width, instructions)
    # Handle three-way interactions like x * z * group
    comp1_info, comp2_info, comp3_info = component_info
    w1, w2, w3 = component_widths
    
    # For now, handle the most common case: scalar * scalar * vector
    if w1 == 1 && w2 == 1 && w3 > 1
        # Two scalars and one vector: x * z * group
        comp1_term = comp1_info[:term]
        comp2_term = comp2_info[:term]
        comp3_term = comp3_info[:term]
        
        # Generate first scalar
        if comp1_term isa Union{ContinuousTerm, Term}
            col1 = comp1_term.sym
            push!(instructions, "@inbounds scalar1 = Float64(data.$col1[row_idx])")
        else
            push!(instructions, "@inbounds scalar1 = 1.0")
        end
        
        # Generate second scalar
        if comp2_term isa Union{ContinuousTerm, Term}
            col2 = comp2_term.sym
            push!(instructions, "@inbounds scalar2 = Float64(data.$col2[row_idx])")
        else
            push!(instructions, "@inbounds scalar2 = 1.0")
        end
        
        # Combine scalars
        push!(instructions, "@inbounds scalar_product = scalar1 * scalar2")
        
        # Handle categorical vector component
        if comp3_term isa CategoricalTerm
            col3 = comp3_term.sym
            contrast_matrix = comp3_term.contrasts.matrix
            n_levels = size(contrast_matrix, 1)
            
            # Get categorical level
            push!(instructions, "@inbounds cat_val = data.$col3[row_idx]")
            push!(instructions, "@inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
            push!(instructions, "@inbounds level_code = clamp(level_code, 1, $n_levels)")
            
            # Generate each contrast column multiplied by scalar product
            for k in 1:w3
                pos = start_pos + k - 1
                values = [contrast_matrix[i, k] for i in 1:n_levels]
                
                if n_levels == 2
                    push!(instructions, "@inbounds contrast_val = level_code == 1 ? $(values[1]) : $(values[2])")
                elseif n_levels == 3
                    push!(instructions, "@inbounds contrast_val = level_code == 1 ? $(values[1]) : level_code == 2 ? $(values[2]) : $(values[3])")
                else
                    ternary_chain = "level_code == 1 ? $(values[1])"
                    for i in 2:(n_levels-1)
                        ternary_chain *= " : level_code == $i ? $(values[i])"
                    end
                    ternary_chain *= " : $(values[n_levels])"
                    push!(instructions, "@inbounds contrast_val = $ternary_chain")
                end
                
                push!(instructions, "@inbounds row_vec[$pos] = scalar_product * contrast_val")
            end
        else
            @warn "Complex three-way interaction component: $comp3_term"
            for k in 1:w3
                pos = start_pos + k - 1
                push!(instructions, "@inbounds row_vec[$pos] = scalar_product")
            end
        end
        
    else
        # Other three-way combinations not implemented
        @warn "Complex three-way interaction: widths $component_widths not implemented"
        for i in 0:(total_width-1)
            push!(instructions, "@inbounds row_vec[$(start_pos + i)] = 1.0  # Three-way fallback")
        end
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

"""
Test the fixed interaction generation on a simple case
"""
function test_interaction_fix()
    println("=== Testing Fixed Interaction Generation ===")
    
    # Create test data with known values
    df = DataFrame(
        x = [2.0, 3.0],
        y = [1.0, 2.0], 
        z = [MathConstants.e, 2.0],
        group = categorical(["B", "A"])
    )
    
    # Test simple interaction: x * group
    formula = @formula(y ~ x + group + x & group)
    model = lm(formula, df)
    
    println("Test data:")
    println("  x = $(df.x[1])")
    println("  group = $(df.group[1]) (should be level 2)")
    
    # Expected model matrix row:
    # [intercept, x, group_contrast1, group_contrast2, x*group_contrast1, x*group_contrast2]
    # [1.0, 2.0, 1.0, 0.0, 2.0*1.0, 2.0*0.0] = [1.0, 2.0, 1.0, 0.0, 2.0, 0.0]
    mm = modelmatrix(model)
    expected = mm[1, :]
    println("Expected model matrix row: $expected")
    
    # Test Phase 1
    analysis = analyze_formula_structure(model)
    println("\nPhase 1 analysis:")
    for (i, term_analysis) in enumerate(analysis.terms)
        println("  Term $i: $(term_analysis.term_type), width $(term_analysis.width), pos $(term_analysis.start_position)")
    end
    
    # Test Phase 2 with fixed interaction generation
    println("\nPhase 2 instruction generation:")
    instructions = String[]
    
    for (i, term_analysis) in enumerate(analysis.terms)
        if term_analysis.term_type == :interaction
            println("  Generating FIXED interaction for term $i...")
            term_instructions = generate_interaction_instructions(term_analysis)
            append!(instructions, term_instructions)
            println("    Generated $(length(term_instructions)) instructions")
        else
            # Use original generation for non-interaction terms
            term_instructions = generate_term_instructions(term_analysis)
            append!(instructions, term_instructions)
        end
    end
    
    println("\nGenerated instructions:")
    for (i, instr) in enumerate(instructions)
        println("  $i: $instr")
    end
    
    return instructions, expected, analysis
end

# function generate_component_evaluation(comp_term, comp_type, var_name, comp_idx)
#     instructions = String[]
    
#     if comp_type <: ContinuousTerm || comp_type <: Term
#         # Continuous variable: comp_val = Float64(data.x[row_idx])
#         col = comp_term.sym
#         push!(instructions, "@inbounds $var_name = Float64(data.$col[row_idx])")
        
#     elseif comp_type <: CategoricalTerm
#         # Categorical variable: take first contrast column only (since width=1 here)
#         col = comp_term.sym
#         contrast_matrix = comp_term.contrasts.matrix
#         n_levels = size(contrast_matrix, 1)
#         values = [contrast_matrix[i, 1] for i in 1:n_levels]  # First column only
        
#         push!(instructions, "@inbounds cat_val_$comp_idx = data.$col[row_idx]")
#         push!(instructions, "@inbounds level_code_$comp_idx = cat_val_$comp_idx isa CategoricalValue ? levelcode(cat_val_$comp_idx) : 1")
#         push!(instructions, "@inbounds level_code_$comp_idx = clamp(level_code_$comp_idx, 1, $n_levels)")
        
#         if n_levels == 2
#             push!(instructions, "@inbounds $var_name = level_code_$comp_idx == 1 ? $(values[1]) : $(values[2])")
#         else
#             # Generate ternary chain or lookup for more levels
#             ternary_chain = "level_code_$comp_idx == 1 ? $(values[1])"
#             for i in 2:(n_levels-1)
#                 ternary_chain *= " : level_code_$comp_idx == $i ? $(values[i])"
#             end
#             ternary_chain *= " : $(values[n_levels])"
#             push!(instructions, "@inbounds $var_name = $ternary_chain")
#         end
        
#     elseif comp_type <: FunctionTerm
#         # Function term: evaluate based on function type
#         append!(instructions, generate_function_component_evaluation(comp_term, var_name, comp_idx))
        
#     else
#         @warn "Unsupported component type in interaction: $comp_type, using fallback"
#         push!(instructions, "@inbounds $var_name = 1.0  # Fallback")
#     end
    
#     return instructions
# end
