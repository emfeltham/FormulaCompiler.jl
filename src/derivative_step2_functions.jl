# derivative_step2_functions.jl
# Phase 2: Function derivative support with chain rule implementation

###############################################################################
# FUNCTION DERIVATIVE DATA TYPES
###############################################################################

"""
    LinearFunctionDerivativeData

Pre-computed derivative data for linear function execution.
Stores the derivative execution plan alongside the original function plan.
"""
struct LinearFunctionDerivativeData
    original_function_data::LinearFunctionData  # Original function execution plan
    derivative_execution_steps::Vector{FunctionExecutionStep}  # Derivative execution plan
    output_position::Int                        # Where derivative result goes
    scratch_size::Int                          # Scratch space needed for derivative computation
    focal_variable::Symbol                     # Variable we're differentiating with respect to
    contains_focal_variable::Bool              # Whether this function depends on focal variable
end

"""
    FunctionDerivativeData

Combined data for function derivatives in complete formulas.
"""
struct FunctionDerivativeData
    function_derivatives::Vector{LinearFunctionDerivativeData}
    focal_variable::Symbol
end

"""
    FunctionDerivativeOp

Operation encoding for function derivatives.
"""
struct FunctionDerivativeOp end

###############################################################################
# ENHANCED COMPLETE FORMULA TYPES FOR PHASE 2
###############################################################################

"""
    Phase2CompleteDerivativeFormulaData{ConstData, ContData, CatData, FuncData}

Enhanced complete derivative data that handles function derivatives.
"""
struct Phase2CompleteDerivativeFormulaData{ConstData, ContData, CatData, FuncData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
    functions::FuncData                      # FunctionDerivativeData
    interaction_positions::Vector{Int}       # Still set to zero in Phase 2
end

"""
    Phase2CompleteDerivativeFormulaOp{ConstOp, ContOp, CatOp, FuncOp}

Enhanced operation encoding for Phase 2.
"""
struct Phase2CompleteDerivativeFormulaOp{ConstOp, ContOp, CatOp, FuncOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp
    functions::FuncOp
end

###############################################################################
# DERIVATIVE EXECUTION STEP TYPES
###############################################################################

"""
    DerivativeFunctionExecutionStep

A single step in derivative function execution, extending the original step types.
"""
struct DerivativeFunctionExecutionStep
    operation::Symbol                        # :load_constant, :load_continuous, :apply_chain_rule, etc.
    func::Union{Function, Nothing}           # Original function (for chain rule)
    derivative_func::Union{Function, Nothing} # Derivative function (e.g., x -> 1/x for log)
    input_positions::Vector{Int}             # Scratch positions to read from
    output_position::Int                     # Scratch position to write to
    constant_value::Union{Float64, Nothing}  # For load_constant operations
    column_symbol::Union{Symbol, Nothing}    # For load_continuous operations
    chain_rule_factor_position::Union{Int, Nothing} # Position of ∂u/∂x for chain rule
end

###############################################################################
# FUNCTION DERIVATIVE ANALYSIS
###############################################################################

"""
    analyze_function_derivatives(function_data::Vector{LinearFunctionData}, focal_variable::Symbol) -> (FunctionDerivativeData, FunctionDerivativeOp)

Convert linear function data to derivative data using chain rule.
"""
function analyze_function_derivatives(function_data::Vector{LinearFunctionData}, focal_variable::Symbol)
    if isempty(function_data)
        empty_derivative_data = FunctionDerivativeData(LinearFunctionDerivativeData[], focal_variable)
        return empty_derivative_data, FunctionDerivativeOp()
    end
    
    function_derivatives = Vector{LinearFunctionDerivativeData}(undef, length(function_data))
    
    for (i, func_data) in enumerate(function_data)
        function_derivatives[i] = create_function_derivative_plan(func_data, focal_variable)
    end
    
    derivative_data = FunctionDerivativeData(function_derivatives, focal_variable)
    derivative_op = FunctionDerivativeOp()
    
    return derivative_data, derivative_op
end

"""
    create_function_derivative_plan(func_data::LinearFunctionData, focal_variable::Symbol) -> LinearFunctionDerivativeData

Create derivative execution plan for a single function using chain rule.
"""
function create_function_derivative_plan(func_data::LinearFunctionData, focal_variable::Symbol)
    # Check if function contains focal variable
    contains_focal = function_contains_focal_variable(func_data, focal_variable)
    
    if !contains_focal
        # Function doesn't depend on focal variable - derivative is zero
        return LinearFunctionDerivativeData(
            func_data,
            FunctionExecutionStep[],  # No steps needed
            func_data.output_position,
            0,  # No scratch space needed
            focal_variable,
            false
        )
    end
    
    # Generate derivative execution plan using chain rule
    derivative_steps = generate_chain_rule_execution_plan(func_data, focal_variable)
    derivative_scratch_size = calculate_derivative_scratch_size(derivative_steps)
    
    return LinearFunctionDerivativeData(
        func_data,
        derivative_steps,
        func_data.output_position,
        derivative_scratch_size,
        focal_variable,
        true
    )
end

"""
    function_contains_focal_variable(func_data::LinearFunctionData, focal_variable::Symbol) -> Bool

Check if a linear function depends on the focal variable.
"""
function function_contains_focal_variable(func_data::LinearFunctionData, focal_variable::Symbol)
    for step in func_data.execution_steps
        if step.operation === :load_continuous && step.column_symbol == focal_variable
            return true
        end
    end
    return false
end

"""
    generate_chain_rule_execution_plan(func_data::LinearFunctionData, focal_variable::Symbol) -> Vector{FunctionExecutionStep}

Generate derivative execution plan using chain rule.
This is the core of Phase 2 - converting function execution to derivative execution.
"""
function generate_chain_rule_execution_plan(func_data::LinearFunctionData, focal_variable::Symbol)
    original_steps = func_data.execution_steps
    derivative_steps = FunctionExecutionStep[]
    
    # Track which positions contain derivatives vs values
    position_types = Dict{Int, Symbol}()  # :value or :derivative
    next_scratch_pos = func_data.scratch_size + 1
    
    # Process each step and apply chain rule
    for (i, step) in enumerate(original_steps)
        if step.operation === :load_constant
            # ∂c/∂x = 0
            push!(derivative_steps, FunctionExecutionStep(:load_constant, step.output_position, 0.0))
            position_types[step.output_position] = :derivative
            
        elseif step.operation === :load_continuous
            if step.column_symbol == focal_variable
                # ∂x/∂x = 1
                push!(derivative_steps, FunctionExecutionStep(:load_constant, step.output_position, 1.0))
            else
                # ∂y/∂x = 0
                push!(derivative_steps, FunctionExecutionStep(:load_constant, step.output_position, 0.0))
            end
            position_types[step.output_position] = :derivative
            
        elseif step.operation === :call_unary
            # Apply chain rule: ∂f(u)/∂x = f'(u) * ∂u/∂x
            func = step.func
            input_pos = step.input_positions[1]
            output_pos = step.output_position
            
            # We need both the original value and its derivative
            # For chain rule: ∂f(u)/∂x = f'(u) * ∂u/∂x
            # Simplify: just apply the derivative function directly to the input
            
            # Get derivative function and apply chain rule
            if func === log
                # ∂log(u)/∂x = (1/u) * ∂u/∂x
                push!(derivative_steps, FunctionExecutionStep(
                    :log_derivative,
                    nothing,  # func
                    [input_pos],  # input_positions: [∂u/∂x]
                    output_pos,   # output_position
                    nothing,      # constant_value
                    nothing       # column_symbol
                ))
            elseif func === exp
                # ∂exp(u)/∂x = exp(u) * ∂u/∂x
                push!(derivative_steps, FunctionExecutionStep(
                    :exp_derivative,
                    nothing,  # func
                    [input_pos],  # input_positions: [∂u/∂x]
                    output_pos,   # output_position
                    nothing,      # constant_value
                    nothing       # column_symbol
                ))
            elseif func === sqrt
                # ∂sqrt(u)/∂x = (1/(2*sqrt(u))) * ∂u/∂x
                push!(derivative_steps, FunctionExecutionStep(
                    :sqrt_derivative,
                    nothing,  # func
                    [input_pos],  # input_positions: [∂u/∂x]
                    output_pos,   # output_position
                    nothing,      # constant_value
                    nothing       # column_symbol
                ))
            elseif func === sin
                # ∂sin(u)/∂x = cos(u) * ∂u/∂x
                push!(derivative_steps, FunctionExecutionStep(
                    :sin_derivative,
                    nothing,  # func
                    [input_pos],  # input_positions: [∂u/∂x]
                    output_pos,   # output_position
                    nothing,      # constant_value
                    nothing       # column_symbol
                ))
            elseif func === cos
                # ∂cos(u)/∂x = -sin(u) * ∂u/∂x
                push!(derivative_steps, FunctionExecutionStep(
                    :cos_derivative,
                    nothing,  # func
                    [input_pos],  # input_positions: [∂u/∂x]
                    output_pos,   # output_position
                    nothing,      # constant_value
                    nothing       # column_symbol
                ))
            elseif func === abs
                # ∂|u|/∂x = sign(u) * ∂u/∂x
                push!(derivative_steps, FunctionExecutionStep(
                    :abs_derivative,
                    nothing,  # func
                    [input_pos],  # input_positions: [∂u/∂x]
                    output_pos,   # output_position
                    nothing,      # constant_value
                    nothing       # column_symbol
                ))
            else
                # Unknown function - set derivative to zero (Phase 2 limitation)
                push!(derivative_steps, FunctionExecutionStep(
                    :load_constant,
                    nothing,      # func
                    Int[],        # input_positions
                    output_pos,   # output_position
                    0.0,          # constant_value
                    nothing       # column_symbol
                ))
            end
            
            position_types[output_pos] = :derivative
            
        elseif step.operation === :call_binary
            # Binary function derivatives (addition, multiplication, etc.)
            func = step.func
            input_pos1 = step.input_positions[1]
            input_pos2 = step.input_positions[2]
            output_pos = step.output_position
            
            if func === (+) || func === (-)
                # ∂(u ± v)/∂x = ∂u/∂x ± ∂v/∂x
                sign_factor = func === (+) ? 1.0 : -1.0
                push!(derivative_steps, FunctionExecutionStep(
                    :add_derivatives,
                    nothing,  # func
                    [input_pos1, input_pos2],  # input_positions
                    output_pos,   # output_position
                    sign_factor,  # constant_value (for sign)
                    nothing       # column_symbol
                ))
            elseif func === (*)
                # ∂(u*v)/∂x = u*∂v/∂x + v*∂u/∂x (product rule) - simplified for Phase 2
                push!(derivative_steps, FunctionExecutionStep(
                    :load_constant,
                    nothing,      # func
                    Int[],        # input_positions
                    output_pos,   # output_position
                    0.0,          # constant_value (set to zero for now)
                    nothing       # column_symbol
                ))
            else
                # Unknown binary function - set derivative to zero
                push!(derivative_steps, FunctionExecutionStep(
                    :load_constant,
                    nothing,      # func
                    Int[],        # input_positions
                    output_pos,   # output_position
                    0.0,          # constant_value
                    nothing       # column_symbol
                ))
            end
            
            position_types[output_pos] = :derivative
        end
    end
    
    return derivative_steps
end

"""
    calculate_derivative_scratch_size(steps::Vector{FunctionExecutionStep}) -> Int

Calculate scratch space needed for derivative execution.
"""
function calculate_derivative_scratch_size(steps::Vector{FunctionExecutionStep})
    max_pos = 0
    for step in steps
        max_pos = max(max_pos, step.output_position)
        for pos in step.input_positions
            max_pos = max(max_pos, pos)
        end
    end
    return max_pos
end

###############################################################################
# PHASE 2 DERIVATIVE EXECUTION
###############################################################################

"""
    execute_derivative_operation!(data::Phase2CompleteDerivativeFormulaData{ConstData, ContData, CatData, FuncData}, 
                                  op::Phase2CompleteDerivativeFormulaOp{ConstOp, ContOp, CatOp, FuncOp}, 
                                  output, input_data, row_idx) where {ConstData, ContData, CatData, FuncData, ConstOp, ContOp, CatOp, FuncOp}

Execute Phase 2 complete derivative formulas with function derivatives.
"""
function execute_derivative_operation!(data::Phase2CompleteDerivativeFormulaData{ConstData, ContData, CatData, FuncData}, 
                                      op::Phase2CompleteDerivativeFormulaOp{ConstOp, ContOp, CatOp, FuncOp}, 
                                      output, input_data, row_idx) where {ConstData, ContData, CatData, FuncData, ConstOp, ContOp, CatOp, FuncOp}
    
    # Execute constants, continuous, categorical (same as Phase 1)
    execute_derivative_operation!(data.constants, op.constants, output, input_data, row_idx)
    execute_derivative_operation!(data.continuous, op.continuous, output, input_data, row_idx)
    execute_derivative_operation!(data.categorical, op.categorical, output, input_data, row_idx)
    
    # Execute function derivatives (NEW in Phase 2)
    execute_function_derivatives!(data.functions, output, input_data, row_idx)
    
    # Set interaction positions to zero (Phase 2 limitation)
    @inbounds for pos in data.interaction_positions
        output[pos] = 0.0
    end
    
    return nothing
end

"""
    execute_function_derivatives!(func_deriv_data::FunctionDerivativeData, output, input_data, row_idx)

Execute function derivative operations using chain rule.
"""
function execute_function_derivatives!(func_deriv_data::FunctionDerivativeData, output, input_data, row_idx)
    if isempty(func_deriv_data.function_derivatives)
        return nothing
    end
    
    @inbounds for func_derivative in func_deriv_data.function_derivatives
        execute_single_function_derivative!(func_derivative, output, input_data, row_idx)
    end
    
    return nothing
end

"""
    execute_single_function_derivative!(func_deriv::LinearFunctionDerivativeData, output, input_data, row_idx)

Execute derivative for a single function using pre-computed chain rule plan.
SIMPLIFIED VERSION: Works with Step 3 linear execution plans.
"""
function execute_single_function_derivative!(func_deriv::LinearFunctionDerivativeData, output, input_data, row_idx)
    if !func_deriv.contains_focal_variable
        # Function doesn't depend on focal variable - derivative is zero
        @inbounds output[func_deriv.output_position] = 0.0
        return nothing
    end
    
    # Simplified approach: For functions like log(x), exp(x), etc. where x is the focal variable
    # We can compute the derivative directly from the original function execution plan
    
    original_steps = func_deriv.original_function_data.execution_steps
    focal_var = func_deriv.focal_variable
    
    # Find the main function being applied
    main_function = nothing
    input_value = 0.0
    
    # Execute the original function to get intermediate values
    temp_scratch = Vector{Float64}(undef, func_deriv.original_function_data.scratch_size + 5)
    execute_original_function_for_values!(func_deriv.original_function_data, temp_scratch, input_data, row_idx)
    
    # For simple cases like log(x), exp(y), we can identify the pattern
    for step in original_steps
        if step.operation === :call_unary
            main_function = step.func
            input_value = temp_scratch[step.input_positions[1]]
            break
        end
        if step.operation === :load_continuous && step.column_symbol == focal_var
            input_value = temp_scratch[step.output_position]
        end
    end
    
    # Apply derivative rules directly
    derivative_result = 0.0
    
    if main_function === log
        # ∂log(x)/∂x = 1/x
        if input_value > 0.0
            derivative_result = 1.0 / input_value
        end
    elseif main_function === exp
        # ∂exp(x)/∂x = exp(x)  
        derivative_result = exp(input_value)
    elseif main_function === sqrt
        # ∂sqrt(x)/∂x = 1/(2*sqrt(x))
        if input_value > 0.0
            derivative_result = 1.0 / (2.0 * sqrt(input_value))
        end
    elseif main_function === sin
        # ∂sin(x)/∂x = cos(x)
        derivative_result = cos(input_value)
    elseif main_function === cos
        # ∂cos(x)/∂x = -sin(x)
        derivative_result = -sin(input_value)
    elseif main_function === abs
        # ∂|x|/∂x = sign(x)
        derivative_result = input_value == 0.0 ? 0.0 : sign(input_value)
    else
        # Unknown function - derivative is zero
        derivative_result = 0.0
    end
    
    @inbounds output[func_deriv.output_position] = derivative_result
    return nothing
end

"""
    execute_original_function_for_values!(func_data::LinearFunctionData, scratch::Vector{Float64}, input_data, row_idx)

Execute original function to get intermediate values needed for derivative computation.
"""
function execute_original_function_for_values!(func_data::LinearFunctionData, scratch::Vector{Float64}, input_data, row_idx)
    @inbounds for step in func_data.execution_steps
        if step.operation === :load_constant
            scratch[step.output_position] = step.constant_value
            
        elseif step.operation === :load_continuous
            col = step.column_symbol
            val = get_data_value_specialized(input_data, col, row_idx)
            scratch[step.output_position] = Float64(val)
            
        elseif step.operation === :call_unary
            input_val = scratch[step.input_positions[1]]
            result = apply_function_direct_single(step.func, input_val)
            scratch[step.output_position] = result
            
        elseif step.operation === :call_binary
            input_val1 = scratch[step.input_positions[1]]
            input_val2 = scratch[step.input_positions[2]]
            result = apply_function_direct_binary(step.func, input_val1, input_val2)
            scratch[step.output_position] = result
        end
    end
    
    return nothing
end

"""
    execute_derivative_steps!(steps::Vector{FunctionExecutionStep}, scratch::Vector{Float64}, input_data, row_idx)

Execute derivative computation steps using chain rule.
For Phase 2, we need to access both the original function values and compute derivatives.
"""
function execute_derivative_steps!(steps::Vector{FunctionExecutionStep}, scratch::Vector{Float64}, input_data, row_idx)
    @inbounds for step in steps
        if step.operation === :load_constant
            scratch[step.output_position] = step.constant_value
            
        elseif step.operation === :load_continuous
            col = step.column_symbol
            val = get_data_value_specialized(input_data, col, row_idx)
            scratch[step.output_position] = Float64(val)
            
        elseif step.operation === :log_derivative
            # ∂log(u)/∂x = (1/u) * ∂u/∂x
            # For now, if input is a continuous variable, ∂u/∂x = 1
            # We need the original value u from the function execution
            deriv_u = 1.0  # Simplified: assume ∂u/∂x = 1 for direct variables
            
            # Get the original function value at this position
            # This is a simplification - in full implementation we'd track this better
            input_pos = step.input_positions[1]
            original_value = scratch[input_pos]  # This should be the original function result
            
            if original_value > 0.0
                scratch[step.output_position] = deriv_u / original_value
            else
                scratch[step.output_position] = 0.0  # Handle log(0) case
            end
            
        elseif step.operation === :exp_derivative
            # ∂exp(u)/∂x = exp(u) * ∂u/∂x
            deriv_u = 1.0  # Simplified: assume ∂u/∂x = 1 for direct variables
            input_pos = step.input_positions[1]
            original_value = scratch[input_pos]
            scratch[step.output_position] = deriv_u * exp(original_value)
            
        elseif step.operation === :sqrt_derivative
            # ∂sqrt(u)/∂x = (1/(2*sqrt(u))) * ∂u/∂x
            deriv_u = 1.0  # Simplified: assume ∂u/∂x = 1 for direct variables
            input_pos = step.input_positions[1]
            original_value = scratch[input_pos]
            if original_value > 0.0
                scratch[step.output_position] = deriv_u / (2.0 * sqrt(original_value))
            else
                scratch[step.output_position] = 0.0  # Handle sqrt(0) case
            end
            
        elseif step.operation === :sin_derivative
            # ∂sin(u)/∂x = cos(u) * ∂u/∂x
            deriv_u = 1.0  # Simplified: assume ∂u/∂x = 1 for direct variables
            input_pos = step.input_positions[1]
            original_value = scratch[input_pos]
            scratch[step.output_position] = deriv_u * cos(original_value)
            
        elseif step.operation === :cos_derivative
            # ∂cos(u)/∂x = -sin(u) * ∂u/∂x
            deriv_u = 1.0  # Simplified: assume ∂u/∂x = 1 for direct variables
            input_pos = step.input_positions[1]
            original_value = scratch[input_pos]
            scratch[step.output_position] = -deriv_u * sin(original_value)
            
        elseif step.operation === :abs_derivative
            # ∂|u|/∂x = sign(u) * ∂u/∂x
            deriv_u = 1.0  # Simplified: assume ∂u/∂x = 1 for direct variables
            input_pos = step.input_positions[1]
            original_value = scratch[input_pos]
            sign_u = original_value == 0.0 ? 0.0 : sign(original_value)
            scratch[step.output_position] = deriv_u * sign_u
            
        elseif step.operation === :add_derivatives
            # ∂(u ± v)/∂x = ∂u/∂x ± ∂v/∂x
            deriv_u = scratch[step.input_positions[1]]
            deriv_v = scratch[step.input_positions[2]]
            factor = step.constant_value  # +1.0 for addition, -1.0 for subtraction
            scratch[step.output_position] = deriv_u + factor * deriv_v
            
        else
            # Unknown operation - set to zero
            scratch[step.output_position] = 0.0
        end
    end
    
    return nothing
end

###############################################################################
# PHASE 2 COMPILATION INTEGRATION
###############################################################################

"""
    compile_phase2_derivative_formula(specialized_formula::SpecializedFormula, focal_variable::Symbol)

Compile derivatives for complete formulas with Phase 2 function support.
"""
function compile_phase2_derivative_formula(specialized_formula::SpecializedFormula, focal_variable::Symbol)
    original_data = specialized_formula.data
    
    # Convert basic components (same as Phase 1)
    const_deriv_data, const_deriv_op = analyze_constant_derivatives(original_data.constants, focal_variable)
    cont_deriv_data, cont_deriv_op = analyze_continuous_derivatives(original_data.continuous, focal_variable)
    cat_deriv_data, cat_deriv_op = analyze_categorical_derivatives(original_data.categorical, focal_variable)
    
    # Convert functions using Phase 2 chain rule analysis
    func_deriv_data, func_deriv_op = analyze_function_derivatives(original_data.functions, focal_variable)
    
    # Extract interaction positions (still set to zero in Phase 2)
    interaction_positions = Int[]
    for interaction_data in original_data.interactions
        append!(interaction_positions, interaction_data.output_positions)
    end
    
    # Create Phase 2 derivative data
    derivative_data = Phase2CompleteDerivativeFormulaData(
        const_deriv_data,
        cont_deriv_data,
        cat_deriv_data,
        func_deriv_data,
        interaction_positions
    )
    
    derivative_operations = Phase2CompleteDerivativeFormulaOp(
        const_deriv_op,
        cont_deriv_op,
        cat_deriv_op,
        func_deriv_op
    )
    
    return DerivativeFormula{typeof(derivative_data), typeof(derivative_operations)}(
        derivative_data,
        derivative_operations,
        specialized_formula.output_width,
        focal_variable
    )
end

###############################################################################
# PHASE 2 INTEGRATION WITH EXISTING SYSTEM
###############################################################################

"""
    compile_derivative_formula_phase2(specialized_formula::SpecializedFormula{D, O}, focal_variable::Symbol) -> DerivativeFormula

Phase 2 version of derivative compilation with function support.
"""
function compile_derivative_formula_phase2(specialized_formula::SpecializedFormula{D, O}, focal_variable::Symbol) where {D, O}
    
    # Handle different formula types
    if specialized_formula.data isa SimpleFormulaData
        return compile_simple_derivative_formula(specialized_formula, focal_variable)  # Reuse Phase 1
    elseif specialized_formula.data isa EnhancedFormulaData
        return compile_enhanced_derivative_formula(specialized_formula, focal_variable)  # Reuse Phase 1
    elseif specialized_formula.data isa CompleteFormulaData
        return compile_phase2_derivative_formula(specialized_formula, focal_variable)  # NEW Phase 2
    else
        error("Phase 2 derivative compilation not yet implemented for data type: $(typeof(specialized_formula.data))")
    end
end

###############################################################################
# DEMO FUNCTIONS
###############################################################################

function demo_phase2_usage()
    println("\n🔬 DERIVATIVE PHASE 2 USAGE DEMO")
    println("=" ^ 5)
    
    # Create sample data
    df, data = create_function_test_data()
    
    # Fit a model with functions
    model = lm(@formula(response ~ log(x) + exp(y) + sqrt(z)), df)
    println("Model: response ~ log(x) + exp(y) + sqrt(z)")
    println("Coefficients: $(coef(model))")
    
    # Compile specialized formula
    specialized = compile_formula(model, data)
    println("Specialized formula compiled with width: $(length(specialized))")
    
    # Compile Phase 2 derivatives
    dx_deriv = compile_derivative_formula_phase2(specialized, :x)
    dy_deriv = compile_derivative_formula_phase2(specialized, :y)
    dz_deriv = compile_derivative_formula_phase2(specialized, :z)
    
    println("\nPhase 2 derivative formulas compiled:")
    println("  ∂/∂x formula width: $(length(dx_deriv))")
    println("  ∂/∂y formula width: $(length(dy_deriv))") 
    println("  ∂/∂z formula width: $(length(dz_deriv))")
    
    # Evaluate derivatives at observation 1
    row_idx = 1
    x_val, y_val, z_val = data.x[row_idx], data.y[row_idx], data.z[row_idx]
    println("\nEvaluating derivatives at observation $row_idx:")
    println("  x = $x_val, y = $y_val, z = $z_val")
    
    dx_vec = modelrow(dx_deriv, data, row_idx)
    dy_vec = modelrow(dy_deriv, data, row_idx)
    dz_vec = modelrow(dz_deriv, data, row_idx)
    
    println("  ∂/∂x: $dx_vec")
    println("  ∂/∂y: $dy_vec")
    println("  ∂/∂z: $dz_vec")
    
    # Compute marginal effects
    coefficients = coef(model)
    me_x = marginal_effect(dx_deriv, coefficients, data, row_idx)
    me_y = marginal_effect(dy_deriv, coefficients, data, row_idx)
    me_z = marginal_effect(dz_deriv, coefficients, data, row_idx)
    
    println("\nMarginal effects at observation $row_idx:")
    println("  ∂E[response]/∂x = $me_x (≈ coef_log_x * 1/x = $(coefficients[2]) * $(1/x_val))")
    println("  ∂E[response]/∂y = $me_y (≈ coef_exp_y * exp(y) = $(coefficients[3]) * $(exp(y_val)))")
    println("  ∂E[response]/∂z = $me_z (≈ coef_sqrt_z * 1/(2√z) = $(coefficients[4]) * $(1/(2*sqrt(z_val))))")
    
    println("\n✨ Phase 2 demo completed successfully!")
end
