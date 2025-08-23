# derivative_modern.jl
# Updated derivative system for the new zero-allocation architecture

###############################################################################
# CORE DERIVATIVE TYPES (UPDATED FOR NEW ARCHITECTURE)
###############################################################################

"""
    DerivativeFormula{DataTuple, OpTuple}

Specialized formula for derivative evaluation.
Mirrors SpecializedFormula structure exactly.
"""
struct DerivativeFormula{DataTuple, OpTuple}
    data::DataTuple
    operations::OpTuple
    output_width::Int
    focal_variable::Symbol
end

# Core call operator - same pattern as SpecializedFormula
function (df::DerivativeFormula{D, O})(
    output::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
) where {D, O}
    execute_operation!(df.data, df.operations, output, data, row_idx)
    return output
end

Base.length(df::DerivativeFormula) = df.output_width

###############################################################################
# DERIVATIVE DATA TYPES (TUPLE-BASED)
###############################################################################

"""
    ContinuousDerivativeData{N, Cols, DerivVals}

Pre-computed derivative data for continuous variables.
DerivVals is compile-time tuple of 1.0/0.0 values.
"""
struct ContinuousDerivativeData{N, Cols, DerivVals}
    columns::Cols                    # NTuple{N, Symbol}
    positions::NTuple{N, Int}
    derivative_values::DerivVals     # NTuple{N, Float64} - compile-time known!
    
    function ContinuousDerivativeData(
        columns::NTuple{N, Symbol}, 
        positions::NTuple{N, Int}, 
        focal_variable::Symbol
    ) where N
        # Pre-compute at compile time which variables get derivative 1.0
        derivative_values = ntuple(N) do i
            columns[i] == focal_variable ? 1.0 : 0.0
        end
        new{N, typeof(columns), typeof(derivative_values)}(
            columns, positions, derivative_values
        )
    end
end

"""
    ConstantDerivativeData{N}

Derivative data for constants (always zero).
"""
struct ConstantDerivativeData{N}
    positions::NTuple{N, Int}
end

"""
    CategoricalDerivativeData{PosTuple}

Derivative data for categoricals (zero for continuous focal).
"""
struct CategoricalDerivativeData{PosTuple}
    positions_tuple::PosTuple  # Tuple of position tuples for each categorical
end

###############################################################################
# FUNCTION DERIVATIVES WITH NEW ARCHITECTURE
###############################################################################

"""
    FunctionDerivativeData{UnaryTuple, IntermediateTuple, FinalTuple}

Function derivatives using the new tuple-based system.
"""
struct FunctionDerivativeData{UnaryTuple, IntermediateTuple, FinalTuple}
    unary_derivatives::UnaryTuple
    intermediate_derivatives::IntermediateTuple  
    final_derivatives::FinalTuple
    focal_variable::Symbol
end

"""
    UnaryFunctionDerivative{F, InputType, DerivType}

Derivative of unary function with compile-time types.
"""
struct UnaryFunctionDerivative{F, InputType, DerivType}
    func::F
    input_source::InputType
    position::Int
    derivative_type::DerivType  # :zero, :identity, :chain_rule
end

###############################################################################
# INTERACTION DERIVATIVES
###############################################################################

"""
    InteractionDerivativeData{InteractionTuple}

Interaction derivatives using product rule.
"""
struct InteractionDerivativeData{InteractionTuple}
    interaction_derivatives::InteractionTuple
    focal_variable::Symbol
end

"""
    SingleInteractionDerivative{C1, C2, Pattern}

Derivative of single interaction using product rule.
"""
struct SingleInteractionDerivative{C1, C2, Pattern}
    component1::C1
    component2::C2
    pattern::Pattern
    output_position::Int
    focal_in_comp1::Bool
    focal_in_comp2::Bool
end

###############################################################################
# COMPLETE DERIVATIVE FORMULA
###############################################################################

"""
    CompleteDerivativeData{ConstData, ContData, CatData, FuncData, IntData}

Complete derivative data matching CompleteFormulaData structure.
"""
struct CompleteDerivativeData{ConstData, ContData, CatData, FuncData, IntData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
    functions::FuncData
    interactions::IntData
    scratch::Vector{Float64}  # Reusable scratch space
end

###############################################################################
# EXECUTION FUNCTIONS (ZERO-ALLOCATION)
###############################################################################

"""
    execute_operation!(
        data::ContinuousDerivativeData{N, Cols, DerivVals},
        op::ContinuousOp{N, Cols},
        output, input_data, row_idx
    ) where {N, Cols, DerivVals}

Execute continuous derivatives - compile-time known values!
"""
function execute_operation!(
    data::ContinuousDerivativeData{N, Cols, DerivVals},
    op::ContinuousOp{N, Cols},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {N, Cols, DerivVals}
    
    @inbounds for i in 1:N
        pos = data.positions[i]
        # Derivative value is compile-time constant!
        output[pos] = data.derivative_values[i]
    end
    
    return nothing
end

"""
    execute_operation!(
        data::ConstantDerivativeData{N},
        op::ConstantOp{N},
        output, input_data, row_idx
    ) where N

Execute constant derivatives (always zero).
"""
function execute_operation!(
    data::ConstantDerivativeData{N},
    op::ConstantOp{N},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where N
    
    @inbounds for i in 1:N
        output[data.positions[i]] = 0.0
    end
    
    return nothing
end

###############################################################################
# FUNCTION DERIVATIVE EXECUTION
###############################################################################

"""
    execute_operation!(
        data::FunctionDerivativeData{UT, IT, FT},
        op::FunctionOp{U, I, F},
        scratch, output, input_data, row_idx
    ) where {UT, IT, FT, U, I, F}

Execute function derivatives using chain rule.
"""
function execute_operation!(
    data::FunctionDerivativeData{UT, IT, FT},
    op::FunctionOp{U, I, F},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {UT, IT, FT, U, I, F}
    
    # Process in dependency order (same as functions)
    execute_intermediate_derivatives_recursive!(
        data.intermediate_derivatives, scratch, output, input_data, row_idx
    )
    execute_unary_derivatives_recursive!(
        data.unary_derivatives, scratch, output, input_data, row_idx
    )
    execute_final_derivatives_recursive!(
        data.final_derivatives, scratch, output, input_data, row_idx
    )
    
    return nothing
end

# Recursive execution helpers (similar pattern to function execution)
function execute_unary_derivatives_recursive!(
    unary_tuple::Tuple{},
    scratch, output, input_data, row_idx
)
    return nothing
end

function execute_unary_derivatives_recursive!(
    unary_tuple::Tuple,
    scratch, output, input_data, row_idx
)
    if length(unary_tuple) > 0
        execute_unary_derivative!(unary_tuple[1], scratch, output, input_data, row_idx)
        if length(unary_tuple) > 1
            execute_unary_derivatives_recursive!(
                Base.tail(unary_tuple), scratch, output, input_data, row_idx
            )
        end
    end
    return nothing
end

"""
    execute_unary_derivative!(
        deriv::UnaryFunctionDerivative{F, InputType, DerivType},
        scratch, output, input_data, row_idx
    ) where {F, InputType, DerivType}

Execute single unary function derivative.
"""
function execute_unary_derivative!(
    deriv::UnaryFunctionDerivative{F, InputType, DerivType},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {F, InputType, DerivType}
    
    if DerivType === :zero
        output[deriv.position] = 0.0
    elseif DerivType === :identity
        output[deriv.position] = 1.0
    else
        # Chain rule case
        input_val = get_input_value_zero_alloc(
            deriv.input_source, output, scratch, input_data, row_idx
        )
        
        # Apply derivative function
        result = if F === typeof(log)
            input_val > 0.0 ? 1.0 / input_val : 0.0
        elseif F === typeof(exp)
            exp(clamp(input_val, -700.0, 700.0))
        elseif F === typeof(sqrt)
            input_val > 0.0 ? 0.5 / sqrt(input_val) : 0.0
        elseif F === typeof(sin)
            cos(input_val)
        elseif F === typeof(cos)
            -sin(input_val)
        elseif F === typeof(abs)
            sign(input_val)
        else
            0.0  # Unknown function
        end
        
        output[deriv.position] = result
    end
    
    return nothing
end

###############################################################################
# INTERACTION DERIVATIVE EXECUTION (PRODUCT RULE)
###############################################################################

"""
    execute_operation!(
        data::InteractionDerivativeData{IT},
        op::InteractionOp{I, F},
        scratch, output, input_data, row_idx
    ) where {IT, I, F}

Execute interaction derivatives using product rule.
"""
function execute_operation!(
    data::InteractionDerivativeData{IT},
    op::InteractionOp{I, F},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {IT, I, F}
    
    execute_interaction_derivatives_recursive!(
        data.interaction_derivatives, scratch, output, input_data, row_idx
    )
    
    return nothing
end

function execute_interaction_derivatives_recursive!(
    int_tuple::Tuple{},
    scratch, output, input_data, row_idx
)
    return nothing
end

function execute_interaction_derivatives_recursive!(
    int_tuple::Tuple,
    scratch, output, input_data, row_idx
)
    if length(int_tuple) > 0
        execute_single_interaction_derivative!(
            int_tuple[1], scratch, output, input_data, row_idx
        )
        if length(int_tuple) > 1
            execute_interaction_derivatives_recursive!(
                Base.tail(int_tuple), scratch, output, input_data, row_idx
            )
        end
    end
    return nothing
end

"""
    execute_single_interaction_derivative!(
        deriv::SingleInteractionDerivative{C1, C2, Pattern},
        scratch, output, input_data, row_idx
    ) where {C1, C2, Pattern}

Execute single interaction derivative using product rule:
∂(u*v)/∂x = (∂u/∂x)*v + u*(∂v/∂x)
"""
function execute_single_interaction_derivative!(
    deriv::SingleInteractionDerivative{C1, C2, Pattern},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {C1, C2, Pattern}
    
    # For each pattern position
    @inbounds for pattern_idx in 1:length(deriv.pattern)
        i, j = deriv.pattern[pattern_idx]
        
        # Get component values
        val1 = get_component_interaction_value(
            deriv.component1, i, input_data, row_idx, output, scratch
        )
        val2 = get_component_interaction_value(
            deriv.component2, j, input_data, row_idx, output, scratch
        )
        
        # Apply product rule
        derivative = 0.0
        if deriv.focal_in_comp1 && !deriv.focal_in_comp2
            # ∂u/∂x * v (only first component has focal)
            derivative = 1.0 * val2  # ∂u/∂x = 1 for focal variable
        elseif !deriv.focal_in_comp1 && deriv.focal_in_comp2
            # u * ∂v/∂x (only second component has focal)
            derivative = val1 * 1.0  # ∂v/∂x = 1 for focal variable
        elseif deriv.focal_in_comp1 && deriv.focal_in_comp2
            # Both have focal (shouldn't happen for single variable)
            derivative = 1.0 * val2 + val1 * 1.0
        else
            # Neither has focal
            derivative = 0.0
        end
        
        output_pos = deriv.output_position + pattern_idx - 1
        output[output_pos] = derivative
    end
    
    return nothing
end

###############################################################################
# COMPILATION FUNCTIONS
###############################################################################

"""
    compile_derivative_formula(
        specialized::SpecializedFormula{D, O},
        focal_variable::Symbol
    ) -> DerivativeFormula

Compile derivative formula from specialized formula.
"""
function compile_derivative_formula(
    specialized::SpecializedFormula{D, O},
    focal_variable::Symbol
) where {D, O}
    
    # Analyze each component for derivatives
    deriv_data, deriv_op = analyze_derivative_components(
        specialized.data, specialized.operations, focal_variable
    )
    
    return DerivativeFormula{typeof(deriv_data), typeof(deriv_op)}(
        deriv_data,
        deriv_op,
        specialized.output_width,
        focal_variable
    )
end

"""
    analyze_derivative_components(data, operations, focal_variable)

Convert formula components to derivative components.
"""
function analyze_derivative_components(data, operations, focal_variable)
    if data isa CompleteFormulaData
        # Handle complete formula with all components
        const_deriv = analyze_constant_derivatives(data.constants, focal_variable)
        cont_deriv = analyze_continuous_derivatives(data.continuous, focal_variable)
        cat_deriv = analyze_categorical_derivatives(data.categorical, focal_variable)
        func_deriv = analyze_function_derivatives(data.functions, operations.functions, focal_variable)
        int_deriv = analyze_interaction_derivatives(data.interactions, operations.interactions, focal_variable)
        
        deriv_data = CompleteDerivativeData(
            const_deriv,
            cont_deriv,
            cat_deriv,
            func_deriv,
            int_deriv,
            Vector{Float64}(undef, max(data.max_function_scratch, data.max_interaction_scratch))
        )
        
        # Operations stay the same type structure
        deriv_op = operations  # Can reuse same operation structure
        
        return deriv_data, deriv_op
    else
        error("Derivative compilation not implemented for $(typeof(data))")
    end
end

"""
    analyze_continuous_derivatives(cont_data::ContinuousData{N, Cols}, focal_variable::Symbol) where {N, Cols}

Convert continuous data to derivative data.
"""
function analyze_continuous_derivatives(
    cont_data::ContinuousData{N, Cols},
    focal_variable::Symbol
) where {N, Cols}
    
    return ContinuousDerivativeData(
        cont_data.columns,
        cont_data.positions,
        focal_variable
    )
end

"""
    analyze_constant_derivatives(const_data::ConstantData{N}, focal_variable::Symbol) where N

Convert constant data to derivative data (always zero).
"""
function analyze_constant_derivatives(
    const_data::ConstantData{N},
    focal_variable::Symbol
) where N
    
    return ConstantDerivativeData{N}(const_data.positions)
end

###############################################################################
# MARGINAL EFFECTS INTERFACE
###############################################################################

"""
    marginal_effect(
        model,
        data::NamedTuple,
        focal_variable::Symbol,
        row_idx::Int
    ) -> Float64

Compute marginal effect at a single observation.
"""
function marginal_effect(
    model,
    data::NamedTuple,
    focal_variable::Symbol,
    row_idx::Int
)
    # Compile formulas
    specialized = compile_formula(model, data)
    derivative = compile_derivative_formula(specialized, focal_variable)
    
    # Get coefficients
    coefficients = coef(model)
    
    # Evaluate derivative
    deriv_vec = Vector{Float64}(undef, length(derivative))
    derivative(deriv_vec, data, row_idx)
    
    # Marginal effect = dot(derivatives, coefficients)
    return dot(deriv_vec, coefficients)
end

"""
    marginal_effects!(
        output::Vector{Float64},
        model,
        data::NamedTuple,
        focal_variables::Vector{Symbol},
        row_idx::Int
    )

Compute marginal effects for multiple variables.
"""
function marginal_effects!(
    output::Vector{Float64},
    model,
    data::NamedTuple,
    focal_variables::Vector{Symbol},
    row_idx::Int
)
    specialized = compile_formula(model, data)
    coefficients = coef(model)
    
    deriv_vec = Vector{Float64}(undef, length(specialized))
    
    for (i, focal_var) in enumerate(focal_variables)
        derivative = compile_derivative_formula(specialized, focal_var)
        derivative(deriv_vec, data, row_idx)
        output[i] = dot(deriv_vec, coefficients)
    end
    
    return output
end

###############################################################################
# AVERAGE MARGINAL EFFECTS
###############################################################################

"""
    average_marginal_effect(
        model,
        data::NamedTuple,
        focal_variable::Symbol
    ) -> Float64

Compute average marginal effect across all observations.
"""
function average_marginal_effect(
    model,
    data::NamedTuple,
    focal_variable::Symbol
)
    n_obs = length(first(data))
    total_effect = 0.0
    
    specialized = compile_formula(model, data)
    derivative = compile_derivative_formula(specialized, focal_variable)
    coefficients = coef(model)
    
    deriv_vec = Vector{Float64}(undef, length(derivative))
    
    for row_idx in 1:n_obs
        derivative(deriv_vec, data, row_idx)
        total_effect += dot(deriv_vec, coefficients)
    end
    
    return total_effect / n_obs
end
