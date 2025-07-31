# derivative_step1_foundation.jl
# Phase 1: Core derivative data types and foundation

###############################################################################
# CORE DERIVATIVE SPECIALIZED TYPES
###############################################################################

"""
    DerivativeFormula{DerivativeDataTuple, DerivativeOpTuple}

Specialized formula for derivative evaluation.
Extends the SpecializedFormula pattern to derivatives.
"""
struct DerivativeFormula{DerivativeDataTuple, DerivativeOpTuple}
    derivative_data::DerivativeDataTuple
    derivative_operations::DerivativeOpTuple
    original_output_width::Int
    focal_variable::Symbol
end

# Core call operator for efficient derivative execution
function (df::DerivativeFormula{D, O})(
    output::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int) where {D, O}
    execute_derivative_operation!(df.derivative_data, df.derivative_operations, output, data, row_idx)
    return output
end

###############################################################################
# DERIVATIVE DATA TYPES
###############################################################################

"""
    ContinuousDerivativeData{N, Cols}

Pre-computed derivative data for continuous variables.
"""
struct ContinuousDerivativeData{N, Cols}
    columns::Cols  # NTuple{N, Symbol} - original columns
    positions::NTuple{N, Int}  # Output positions 
    focal_variable::Symbol  # Runtime constant
    derivative_values::NTuple{N, Float64}  # Pre-computed: 1.0 if col==focal, 0.0 otherwise
    
    function ContinuousDerivativeData(columns::NTuple{N, Symbol}, 
                                     positions::NTuple{N, Int}, 
                                     focal_variable::Symbol) where N
        # Pre-compute derivative values at compilation time
        derivative_values = ntuple(N) do i
            columns[i] == focal_variable ? 1.0 : 0.0
        end
        
        new{N, typeof(columns)}(columns, positions, focal_variable, derivative_values)
    end
end

"""
    ConstantDerivativeData{N}

Pre-computed derivative data for constants (always zero).
"""
struct ConstantDerivativeData{N}
    positions::NTuple{N, Int}   # Output positions
    focal_variable::Symbol      # Runtime constant
    
    function ConstantDerivativeData(positions::NTuple{N, Int}, focal_variable::Symbol) where N
        new{N}(positions, focal_variable)
    end
end

"""
    CategoricalDerivativeData

Pre-computed derivative data for categorical variables (always zero for continuous focal variables).
"""
struct CategoricalDerivativeData
    positions::Vector{Int}   # Output positions where derivatives go
    focal_variable::Symbol   # Runtime constant
    
    function CategoricalDerivativeData(positions::Vector{Int}, focal_variable::Symbol)
        new(positions, focal_variable)
    end
end

"""
    SimpleDerivativeFormulaData{ConstData, ContData, CatData}

Combined derivative data for simple formulas with constants, continuous, and categorical variables.
"""
struct SimpleDerivativeFormulaData{ConstData, ContData, CatData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
end

"""
    CompleteDerivativeFormulaData{ConstData, ContData, CatData}

Derivative data for complete formulas. Functions and interactions set to zero in Phase 1.
"""
struct CompleteDerivativeFormulaData{ConstData, ContData, CatData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
    function_positions::Vector{Int}      # Positions where function terms should be zero
    interaction_positions::Vector{Int}   # Positions where interaction terms should be zero
end

###############################################################################
# DERIVATIVE OPERATION TYPES
###############################################################################

"""
    ContinuousDerivativeOp{N, Cols}

Compile-time encoding of continuous variable derivative operations.
"""
struct ContinuousDerivativeOp{N, Cols}
    function ContinuousDerivativeOp(::ContinuousDerivativeData{N, Cols}) where {N, Cols}
        new{N, Cols}()
    end
end

"""
    ConstantDerivativeOp{N}

Compile-time encoding of constant derivative operations (always zero).
"""
struct ConstantDerivativeOp{N}
    function ConstantDerivativeOp(::ConstantDerivativeData{N}) where {N}
        new{N}()
    end
end

"""
    CategoricalDerivativeOp

Compile-time encoding of categorical derivative operations (always zero for continuous focal).
"""
struct CategoricalDerivativeOp
    function CategoricalDerivativeOp()
        new()
    end
end

"""
    SimpleDerivativeFormulaOp{ConstOp, ContOp, CatOp}

Combined derivative operation encoding for simple formulas.
"""
struct SimpleDerivativeFormulaOp{ConstOp, ContOp, CatOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp
end

"""
    CompleteDerivativeFormulaOp{ConstOp, ContOp, CatOp}

Operation encoding for complete derivative formulas.
"""
struct CompleteDerivativeFormulaOp{ConstOp, ContOp, CatOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp
end

###############################################################################
# DERIVATIVE ANALYSIS FUNCTIONS
###############################################################################

"""
    analyze_continuous_derivatives(data::ContinuousData{N, Cols}, focal_variable::Symbol) -> (ContinuousDerivativeData, ContinuousDerivativeOp)

Convert continuous variable data to derivative data.
"""
function analyze_continuous_derivatives(data::ContinuousData{N, Cols}, focal_variable::Symbol) where {N, Cols}
    if N == 0
        # No continuous operations
        empty_derivative_data = ContinuousDerivativeData((), (), focal_variable)
        return empty_derivative_data, ContinuousDerivativeOp(empty_derivative_data)
    end
    
    derivative_data = ContinuousDerivativeData(data.columns, data.positions, focal_variable)
    derivative_op = ContinuousDerivativeOp(derivative_data)
    
    return derivative_data, derivative_op
end

"""
    analyze_constant_derivatives(data::ConstantData{N}, focal_variable::Symbol) -> (ConstantDerivativeData, ConstantDerivativeOp)

Convert constant data to derivative data (always zero).
"""
function analyze_constant_derivatives(data::ConstantData{N}, focal_variable::Symbol) where N
    if N == 0
        # No constant operations
        empty_derivative_data = ConstantDerivativeData((), focal_variable)
        return empty_derivative_data, ConstantDerivativeOp(empty_derivative_data)
    end
    
    derivative_data = ConstantDerivativeData(data.positions, focal_variable)
    derivative_op = ConstantDerivativeOp(derivative_data)
    
    return derivative_data, derivative_op
end

"""
    analyze_categorical_derivatives(categorical_data::Vector{CategoricalData}, focal_variable::Symbol) -> (Vector{CategoricalDerivativeData}, CategoricalDerivativeOp)

Convert categorical data to derivative data (always zero for continuous focal variables).
"""
function analyze_categorical_derivatives(categorical_data::Vector{CategoricalData}, focal_variable::Symbol)
    if isempty(categorical_data)
        return CategoricalDerivativeData[], CategoricalDerivativeOp()
    end
    
    derivative_data_vec = Vector{CategoricalDerivativeData}(undef, length(categorical_data))
    
    for (i, cat_data) in enumerate(categorical_data)
        derivative_data_vec[i] = CategoricalDerivativeData(copy(cat_data.positions), focal_variable)
    end
    
    derivative_op = CategoricalDerivativeOp()
    
    return derivative_data_vec, derivative_op
end

###############################################################################
# DERIVATIVE EXECUTION FUNCTIONS
###############################################################################

"""
    execute_derivative_operation!(data::ContinuousDerivativeData{N, Cols}, 
                                  op::ContinuousDerivativeOp{N, Cols}, 
                                  output, input_data, row_idx) where {N, Cols}

Execute continuous variable derivatives.
"""
function execute_derivative_operation!(data::ContinuousDerivativeData{N, Cols}, 
                                      op::ContinuousDerivativeOp{N, Cols}, 
                                      output, input_data, row_idx) where {N, Cols}
    
    @inbounds for i in 1:N
        pos = data.positions[i]
        derivative_val = data.derivative_values[i]  # Pre-computed: 1.0 or 0.0
        output[pos] = derivative_val
    end
    
    return nothing
end

"""
    execute_derivative_operation!(data::ConstantDerivativeData{N}, 
                                  op::ConstantDerivativeOp{N}, 
                                  output, input_data, row_idx) where {N}

Execute constant derivatives (always zero).
"""
function execute_derivative_operation!(data::ConstantDerivativeData{N}, 
                                      op::ConstantDerivativeOp{N}, 
                                      output, input_data, row_idx) where {N}
    
    @inbounds for i in 1:N
        pos = data.positions[i]
        output[pos] = 0.0  # ∂c/∂x = 0
    end
    
    return nothing
end

"""
    execute_derivative_operation!(categorical_data::Vector{CategoricalDerivativeData}, 
                                  op::CategoricalDerivativeOp, 
                                  output, input_data, row_idx)

Execute categorical derivatives (always zero for continuous focal variables).
"""
function execute_derivative_operation!(categorical_data::Vector{CategoricalDerivativeData}, 
                                      op::CategoricalDerivativeOp, 
                                      output, input_data, row_idx)
    
    # All categorical derivatives w.r.t. continuous variables are zero
    @inbounds for cat_data in categorical_data
        for pos in cat_data.positions
            output[pos] = 0.0
        end
    end
    
    return nothing
end

"""
    execute_derivative_operation!(data::SimpleDerivativeFormulaData{ConstData, ContData, CatData}, 
                                  op::SimpleDerivativeFormulaOp{ConstOp, ContOp, CatOp}, 
                                  output, input_data, row_idx) where {ConstData, ContData, CatData, ConstOp, ContOp, CatOp}

Execute simple derivative formulas with constants, continuous, and categorical variables.
"""
function execute_derivative_operation!(data::SimpleDerivativeFormulaData{ConstData, ContData, CatData}, 
                                      op::SimpleDerivativeFormulaOp{ConstOp, ContOp, CatOp}, 
                                      output, input_data, row_idx) where {ConstData, ContData, CatData, ConstOp, ContOp, CatOp}
    
    # Execute constant derivatives (always zero)
    execute_derivative_operation!(data.constants, op.constants, output, input_data, row_idx)
    
    # Execute continuous variable derivatives
    execute_derivative_operation!(data.continuous, op.continuous, output, input_data, row_idx)
    
    # Execute categorical variable derivatives (always zero for continuous focal)
    execute_derivative_operation!(data.categorical, op.categorical, output, input_data, row_idx)
    
    return nothing
end

"""
    execute_derivative_operation!(data::CompleteDerivativeFormulaData{ConstData, ContData, CatData}, 
                                  op::CompleteDerivativeFormulaOp{ConstOp, ContOp, CatOp}, 
                                  output, input_data, row_idx) where {ConstData, ContData, CatData, ConstOp, ContOp, CatOp}

Execute complete derivative formulas.

N.B., Functions and interactions are set to zero in Phase 1.
"""
function execute_derivative_operation!(data::CompleteDerivativeFormulaData{ConstData, ContData, CatData}, 
                                      op::CompleteDerivativeFormulaOp{ConstOp, ContOp, CatOp}, 
                                      output, input_data, row_idx) where {ConstData, ContData, CatData, ConstOp, ContOp, CatOp}
    
    # Execute constant derivatives (always zero)
    execute_derivative_operation!(data.constants, op.constants, output, input_data, row_idx)
    
    # Execute continuous variable derivatives
    execute_derivative_operation!(data.continuous, op.continuous, output, input_data, row_idx)
    
    # Execute categorical variable derivatives (always zero for continuous focal)
    execute_derivative_operation!(data.categorical, op.categorical, output, input_data, row_idx)
    
    # Set function positions to zero (Phase 1 limitation)
    @inbounds for pos in data.function_positions
        output[pos] = 0.0
    end
    
    # Set interaction positions to zero (Phase 1 limitation)
    @inbounds for pos in data.interaction_positions
        output[pos] = 0.0
    end
    
    return nothing
end

###############################################################################
# DERIVATIVE COMPILATION FROM SPECIALIZED FORMULAS
###############################################################################

"""
    compile_derivative_formula(specialized_formula::SpecializedFormula{D, O}, focal_variable::Symbol) -> DerivativeFormula

Convert a specialized formula to its derivative with respect to focal_variable.
"""
function compile_derivative_formula(specialized_formula::SpecializedFormula{D, O}, focal_variable::Symbol) where {D, O}
    
    # Handle simple formulas (Steps 1-2 only)
    if specialized_formula.data isa SimpleFormulaData
        return compile_simple_derivative_formula(specialized_formula, focal_variable)
    elseif specialized_formula.data isa EnhancedFormulaData
        return compile_enhanced_derivative_formula(specialized_formula, focal_variable)
    elseif specialized_formula.data isa CompleteFormulaData
        return compile_complete_derivative_formula(specialized_formula, focal_variable)
    else
        error("Derivative compilation not yet implemented for data type: $(typeof(specialized_formula.data))")
    end
end

"""
    compile_simple_derivative_formula(specialized_formula::SpecializedFormula, focal_variable::Symbol)

Compile derivatives for simple formulas (constants + continuous variables only).
"""
function compile_simple_derivative_formula(specialized_formula::SpecializedFormula, focal_variable::Symbol)
    original_data = specialized_formula.data
    
    # Convert each component to derivative form
    const_deriv_data, const_deriv_op = analyze_constant_derivatives(original_data.constants, focal_variable)
    cont_deriv_data, cont_deriv_op = analyze_continuous_derivatives(original_data.continuous, focal_variable)
    
    # Create empty categorical derivative data for simple formulas
    empty_cat_deriv_data, empty_cat_deriv_op = analyze_categorical_derivatives(CategoricalData[], focal_variable)
    
    # Combine into derivative data
    derivative_data = SimpleDerivativeFormulaData(
        const_deriv_data,
        cont_deriv_data,
        empty_cat_deriv_data
    )
    
    derivative_operations = SimpleDerivativeFormulaOp(
        const_deriv_op,
        cont_deriv_op,
        empty_cat_deriv_op
    )
    
    return DerivativeFormula{typeof(derivative_data), typeof(derivative_operations)}(
        derivative_data,
        derivative_operations,
        specialized_formula.output_width,
        focal_variable
    )
end

"""
    compile_enhanced_derivative_formula(specialized_formula::SpecializedFormula, focal_variable::Symbol)

Compile derivatives for enhanced formulas (constants + continuous + categorical).
"""
function compile_enhanced_derivative_formula(specialized_formula::SpecializedFormula, focal_variable::Symbol)
    original_data = specialized_formula.data
    
    # Convert each component to derivative form
    const_deriv_data, const_deriv_op = analyze_constant_derivatives(original_data.constants, focal_variable)
    cont_deriv_data, cont_deriv_op = analyze_continuous_derivatives(original_data.continuous, focal_variable)
    cat_deriv_data, cat_deriv_op = analyze_categorical_derivatives(original_data.categorical, focal_variable)
    
    # Combine into derivative data
    derivative_data = SimpleDerivativeFormulaData(
        const_deriv_data,
        cont_deriv_data,
        cat_deriv_data
    )
    
    derivative_operations = SimpleDerivativeFormulaOp(
        const_deriv_op,
        cont_deriv_op,
        cat_deriv_op
    )
    
    return DerivativeFormula{typeof(derivative_data), typeof(derivative_operations)}(
        derivative_data,
        derivative_operations,
        specialized_formula.output_width,
        focal_variable
    )
end

"""
    compile_complete_derivative_formula(specialized_formula::SpecializedFormula, focal_variable::Symbol)

Compile derivatives for complete formulas (constants + continuous + categorical + functions + interactions).
For Phase 1, we handle constants, continuous, and categorical. Functions and interactions return zero derivatives.
"""
function compile_complete_derivative_formula(specialized_formula::SpecializedFormula, focal_variable::Symbol)
    original_data = specialized_formula.data
    
    # Convert basic components to derivative form
    const_deriv_data, const_deriv_op = analyze_constant_derivatives(original_data.constants, focal_variable)
    cont_deriv_data, cont_deriv_op = analyze_continuous_derivatives(original_data.continuous, focal_variable)
    cat_deriv_data, cat_deriv_op = analyze_categorical_derivatives(original_data.categorical, focal_variable)
    
    # Collect positions where functions and interactions should be zero
    function_positions = Int[]
    interaction_positions = Int[]
    
    # Extract function positions
    for func_data in original_data.functions
        push!(function_positions, func_data.output_position)
    end
    
    # Extract interaction positions  
    for interaction_data in original_data.interactions
        append!(interaction_positions, interaction_data.output_positions)
    end
    
    # Create complete derivative data
    derivative_data = CompleteDerivativeFormulaData(
        const_deriv_data,
        cont_deriv_data,
        cat_deriv_data,
        function_positions,
        interaction_positions
    )
    
    derivative_operations = CompleteDerivativeFormulaOp(
        const_deriv_op,
        cont_deriv_op,
        cat_deriv_op
    )
    
    return DerivativeFormula{typeof(derivative_data), typeof(derivative_operations)}(
        derivative_data,
        derivative_operations,
        specialized_formula.output_width,
        focal_variable
    )
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

"""
    Base.length(df::DerivativeFormula) -> Int

Get the output width of a derivative formula (same as original formula).
"""
Base.length(df::DerivativeFormula) = df.original_output_width

"""
    show_derivative_info(df::DerivativeFormula)

Display information about a derivative formula.
"""
function show_derivative_info(df::DerivativeFormula{D, O}) where {D, O}
    println("DerivativeFormula Information:")
    println("  Derivative data type: $D")
    println("  Derivative operation type: $O") 
    println("  Output width: $(df.original_output_width)")
    println("  Focal variable: $(df.focal_variable)")
    
    if df.derivative_data isa SimpleDerivativeFormulaData
        # Show which variables have non-zero derivatives
        cont_data = df.derivative_data.continuous
        if !isempty(cont_data.columns)
            println("  Variables with non-zero derivatives:")
            for (i, col) in enumerate(cont_data.columns)
                deriv_val = cont_data.derivative_values[i]
                if deriv_val != 0.0
                    println("    ∂/∂$(df.focal_variable) of $col = $deriv_val")
                end
            end
        end
    elseif df.derivative_data isa CompleteDerivativeFormulaData
        # Show complete formula derivative info
        cont_data = df.derivative_data.continuous
        if !isempty(cont_data.columns)
            println("  Variables with non-zero derivatives:")
            for (i, col) in enumerate(cont_data.columns)
                deriv_val = cont_data.derivative_values[i]
                if deriv_val != 0.0
                    println("    ∂/∂$(df.focal_variable) of $col = $deriv_val")
                end
            end
        end
        
        println("  Phase 1 limitations:")
        if !isempty(df.derivative_data.function_positions)
            println("    Function derivatives set to zero at positions: $(df.derivative_data.function_positions)")
        end
        if !isempty(df.derivative_data.interaction_positions)
            println("    Interaction derivatives set to zero at positions: $(df.derivative_data.interaction_positions)")
        end
    end
end

###############################################################################
# INTEGRATION WITH EXISTING MODELROW SYSTEM
###############################################################################

"""
    modelrow!(row_vec, derivative_formula::DerivativeFormula, data, row_idx)

Evaluate derivative model row using derivative formula.
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    derivative_formula::DerivativeFormula{D, O}, 
    data, 
    row_idx::Int
) where {D, O}
    @assert length(row_vec) >= length(derivative_formula) "Vector too small: need $(length(derivative_formula)), got $(length(row_vec))"
    @assert 1 <= row_idx <= length(first(data)) "Invalid row index: $row_idx (data has $(length(first(data))) rows)"

    derivative_formula(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow(derivative_formula::DerivativeFormula, data, row_idx) -> Vector{Float64}

Evaluate derivative model row (allocating version).
"""
function modelrow(
    derivative_formula::DerivativeFormula{D, O}, 
    data, 
    row_idx::Int
) where {D, O}
    row_vec = Vector{Float64}(undef, length(derivative_formula))
    derivative_formula(row_vec, data, row_idx)
    return row_vec
end

###############################################################################
# MARGINAL EFFECTS COMPUTATION
###############################################################################

"""
    marginal_effect(derivative_formula::DerivativeFormula, coefficients::Vector{Float64}, data, row_idx::Int) -> Float64

Compute marginal effect: dot(derivative_vector, coefficients).
"""
function marginal_effect(derivative_formula::DerivativeFormula, 
                        coefficients::Vector{Float64}, 
                        data, 
                        row_idx::Int)
    
    @assert length(coefficients) == length(derivative_formula) "Coefficient length mismatch"
    
    # Evaluate derivative
    deriv_vec = Vector{Float64}(undef, length(derivative_formula))
    derivative_formula(deriv_vec, data, row_idx)
    
    # Compute marginal effect
    return dot(deriv_vec, coefficients)
end

"""
    marginal_effect!(output_vec, derivative_formulas::Vector{DerivativeFormula}, 
                     coefficients::Vector{Float64}, data, row_idx::Int)

Compute marginal effects for multiple variables (zero allocations for output).
"""
function marginal_effect!(output_vec::Vector{Float64},
                         derivative_formulas::Vector{<:DerivativeFormula}, 
                         coefficients::Vector{Float64}, 
                         data, 
                         row_idx::Int)
    
    @assert length(output_vec) >= length(derivative_formulas) "Output vector too small"
    
    # Pre-allocate derivative vector (reused across variables)
    deriv_vec = Vector{Float64}(undef, length(coefficients))
    
    @inbounds for (i, deriv_formula) in enumerate(derivative_formulas)
        # Evaluate derivative for this variable
        deriv_formula(deriv_vec, data, row_idx)
        
        # Compute marginal effect
        output_vec[i] = dot(deriv_vec, coefficients)
    end
    
    return output_vec
end

###############################################################################
# DEMO FUNCTIONS
###############################################################################

function demo_derivative_usage()
    println("\n🔬 DERIVATIVE PHASE 1 USAGE DEMO")
    println("=" ^ 5)
    
    # Create sample data
    df, data = create_test_data()
    
    # Fit a simple model
    model = lm(@formula(response ~ x + y + z), df)
    println("Model: response ~ x + y + z")
    println("Coefficients: $(coef(model))")
    
    # Compile specialized formula
    specialized = compile_formula_specialized(model, data)
    println("Specialized formula compiled with width: $(length(specialized))")
    
    # Compile derivatives for each variable
    dx_deriv = compile_derivative_formula(specialized, :x)
    dy_deriv = compile_derivative_formula(specialized, :y)
    dz_deriv = compile_derivative_formula(specialized, :z)
    
    println("\nDerivative formulas compiled:")
    println("  ∂/∂x formula width: $(length(dx_deriv))")
    println("  ∂/∂y formula width: $(length(dy_deriv))") 
    println("  ∂/∂z formula width: $(length(dz_deriv))")
    
    # Evaluate derivatives at observation 1
    row_idx = 1
    println("\nEvaluating derivatives at observation $row_idx:")
    
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
    println("  ∂E[response]/∂x = $me_x")
    println("  ∂E[response]/∂y = $me_y")
    println("  ∂E[response]/∂z = $me_z")
    
    # Batch marginal effects
    derivative_formulas = [dx_deriv, dy_deriv, dz_deriv]
    me_vec = Vector{Float64}(undef, 3)
    marginal_effect!(me_vec, derivative_formulas, coefficients, data, row_idx)
    
    println("\nBatch marginal effects: $me_vec")
    
    println("\n✨ Demo completed successfully!")
end
