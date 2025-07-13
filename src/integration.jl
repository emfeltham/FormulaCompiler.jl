
# STEP-BY-STEP INTEGRATION

###############################################################################
# 1. Create new file: src/compositional_compiler.jl
###############################################################################

# Copy the entire allocation_efficient_compositional.jl content into this file
# (The ExpressionNode types, CompilationContext, parsing functions, etc.)

###############################################################################
# 2. Update EfficientModelMatrices.jl - Add these lines:
###############################################################################

# In EfficientModelMatrices.jl, add after existing includes:
include("compositional_compiler.jl")

# Add to exports:
export compile_formula_compositional_efficient, CompilationContext

###############################################################################
# 3. Update compiled_formula_generated.jl - Replace backend
###############################################################################

# REPLACE the existing compile_to_instructions function:
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

###############################################################################
# 4. Update compile_formula_generated function
###############################################################################

# REPLACE the existing function in compiled_formula_generated.jl:
function compile_formula_generated(model)
    # Use compositional compiler
    func_name, output_width, column_names = compile_formula_compositional_efficient(model)
    
    # Register in cache for @generated dispatch
    formula_hash = hash(string(fixed_effects_form(model).rhs))
    dummy_instructions = [LoadConstant(1, 1.0)]  # Placeholder for compatibility
    FORMULA_CACHE[formula_hash] = (dummy_instructions, column_names, output_width)
    
    return (Val(formula_hash), output_width, column_names)
end

###############################################################################
# 5. Create fallback interface (Optional but recommended)
###############################################################################



###############################################################################
# 6. User Interface Update
###############################################################################

# Your users would now call:
# OLD: compile_formula_generated(model)  
# NEW: compile_formula_compositional_efficient(model)  # Direct
# OR:  compile_formula_auto(model)                     # Auto-dispatch

# The interface is:
func_name, output_width, column_names = compile_formula_compositional_efficient(model)
row_vec = Vector{Float64}(undef, output_width)
data = Tables.columntable(df)

# Direct function call (fastest):
compiled_func = getproperty(Main, func_name)
compiled_func(row_vec, data, row_idx)

# OR through @generated interface (if you keep it):
formula_val, output_width, column_names = compile_formula_generated(model)
modelrow!(row_vec, formula_val, data, row_idx)

###############################################################################
# 7. Testing Integration
###############################################################################

# Test file to verify everything works:
function test_integration()
    using DataFrames, GLM, CategoricalArrays
    
    # Test data
    n = 100
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B"], n))
    )
    
    # Test the problematic formula that failed before
    model = lm(@formula(y ~ x + x^2 * log(z) + group), df)
    
    println("Testing compositional compiler...")
    func_name, output_width, column_names = compile_formula_compositional_efficient(model)
    
    println("✅ Compilation successful!")
    println("Function: $func_name")
    println("Output width: $output_width")
    println("Columns: $column_names")
    
    # Test evaluation
    row_vec = Vector{Float64}(undef, output_width)
    data = Tables.columntable(df)
    compiled_func = getproperty(Main, func_name)
    
    compiled_func(row_vec, data, 1)
    println("✅ Evaluation successful!")
    
    # Verify correctness
    mm_row = modelmatrix(model)[1, :]
    println("Generated: ", row_vec)
    println("Expected:  ", mm_row)
    println("Match: ", isapprox(mm_row, row_vec, atol=1e-10))
    
    return func_name, output_width, column_names
end

###############################################################################
# 8. Migration Guide for Existing Code
###############################################################################

# If you have existing code using compile_formula_generated:

# OLD CODE:
# formula_val, output_width, column_names = compile_formula_generated(model)
# row_vec = Vector{Float64}(undef, output_width)
# @btime modelrow!(row_vec, formula_val, data, 1)

# NEW CODE (Option 1 - Direct):
# func_name, output_width, column_names = compile_formula_compositional_efficient(model)
# row_vec = Vector{Float64}(undef, output_width)
# compiled_func = getproperty(Main, func_name)
# @btime compiled_func(row_vec, data, 1)

# NEW CODE (Option 2 - Keep @generated interface):
# Just update the backend, existing code still works but now handles complex formulas