# step1_specialized_core.jl
# Core foundation for specialized formula execution with continuous variables only

###############################################################################
# CORE SPECIALIZED TYPES
###############################################################################

"""
    SpecializedFormula{DataTuple, OpTuple}

Universal specialized formula executor. DataTuple contains all pre-computed data,
OpTuple encodes the operation structure at the type level.
"""
struct SpecializedFormula{DataTuple, OpTuple}
    data::DataTuple
    operations::OpTuple
    output_width::Int
end

# Core call operator for execution
function (sf::SpecializedFormula{D, O})(
    output::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int) where {D, O}
    execute_specialized!(sf.data, sf.operations, output, data, row_idx)
    return output
end

###############################################################################
# DATA TYPES FOR STEP 1
###############################################################################

"""
    ContinuousData{N, Cols}

Pre-computed data for continuous variables. N is the number of variables,
Cols is a compile-time tuple of column symbols.
"""
struct ContinuousData{N, Cols}
    columns::Cols  # NTuple{N, Symbol} - compile-time known columns
    positions::NTuple{N, Int}  # Output positions for each column
    
    function ContinuousData(columns::NTuple{N, Symbol}, positions::NTuple{N, Int}) where N
        new{N, typeof(columns)}(columns, positions)
    end
end

"""
    ConstantData{N}

Pre-computed data for constant operations.
"""
struct ConstantData{N}
    values::NTuple{N, Float64}  # Constant values
    positions::NTuple{N, Int}   # Output positions
    
    function ConstantData(values::NTuple{N, Float64}, positions::NTuple{N, Int}) where N
        new{N}(values, positions)
    end
end

###############################################################################
# OPERATION TYPES FOR STEP 1
###############################################################################

"""
    ContinuousOp{N, Cols}

Compile-time encoding of continuous variable operations.
"""
struct ContinuousOp{N, Cols}
    function ContinuousOp(::ContinuousData{N, Cols}) where {N, Cols}
        new{N, Cols}()
    end
end

"""
    ConstantOp{N}

Compile-time encoding of constant operations.
"""
struct ConstantOp{N}
    function ConstantOp(::ConstantData{N}) where N
        new{N}()
    end
end

"""
    SimpleFormulaOp{ConstOp, ContOp}

Combined operation encoding for simple formulas.
"""
struct SimpleFormulaOp{ConstOp, ContOp}
    constants::ConstOp
    continuous::ContOp
end

###############################################################################
# ANALYSIS FUNCTIONS
###############################################################################

"""
    analyze_constant_operations(evaluator::CombinedEvaluator) -> (ConstantData, ConstantOp)

Extract constant data from a CombinedEvaluator's constant operations.
"""
function analyze_constant_operations(evaluator::CombinedEvaluator)
    constant_ops = evaluator.constant_ops
    n_ops = length(constant_ops)
    
    if n_ops == 0
        # No constant operations
        empty_data = ConstantData((), ())
        return empty_data, ConstantOp(empty_data)
    end
    
    # Extract values and positions
    values = ntuple(n_ops) do i
        constant_ops[i].value
    end
    
    positions = ntuple(n_ops) do i
        constant_ops[i].position
    end
    
    constant_data = ConstantData(values, positions)
    operation = ConstantOp(constant_data)
    
    return constant_data, operation
end

"""
    analyze_continuous_operations(evaluator::CombinedEvaluator) -> (ContinuousData, ContinuousOp)

Extract continuous variable data from a CombinedEvaluator's continuous operations.
"""
function analyze_continuous_operations(evaluator::CombinedEvaluator)
    continuous_ops = evaluator.continuous_ops
    n_ops = length(continuous_ops)
    
    if n_ops == 0
        # No continuous operations
        empty_data = ContinuousData((), ())
        return empty_data, ContinuousOp(empty_data)
    end
    
    # Extract columns from type parameters and positions
    columns = ntuple(n_ops) do i
        typeof(continuous_ops[i]).parameters[1]
    end
    
    positions = ntuple(n_ops) do i
        continuous_ops[i].position
    end
    
    continuous_data = ContinuousData(columns, positions)
    operation = ContinuousOp(continuous_data)
    
    return continuous_data, operation
end

###############################################################################
# EXECUTION FUNCTIONS
###############################################################################

"""
    execute_specialized!(data, operations, output, input_data, row_idx)

Type-stable execution dispatcher for specialized formulas.
"""
function execute_specialized!(data, operations, output, input_data, row_idx)
    execute_operation!(data, operations, output, input_data, row_idx)
end

"""
    execute_operation!(data::ContinuousData{N, Cols}, op::ContinuousOp{N, Cols}, 
                      output, input_data, row_idx) where {N, Cols}

Execute continuous variable operations.
"""
# Specialized zero-allocation methods for common cases
function execute_operation!(data::ContinuousData{1, Cols}, op::ContinuousOp{1, Cols}, 
                           output, input_data, row_idx) where {Cols}
    @inbounds begin
        col = data.columns[1]
        pos = data.positions[1]
        val = get_data_value_type_stable(input_data, Val{col}(), row_idx)
        output[pos] = Float64(val)
    end
    return nothing
end

function execute_operation!(data::ContinuousData{2, Cols}, op::ContinuousOp{2, Cols}, 
                           output, input_data, row_idx) where {Cols}
    @inbounds begin
        # First variable
        col1 = data.columns[1]
        pos1 = data.positions[1]
        val1 = get_data_value_type_stable(input_data, Val{col1}(), row_idx)
        output[pos1] = Float64(val1)
        
        # Second variable  
        col2 = data.columns[2]
        pos2 = data.positions[2]
        val2 = get_data_value_type_stable(input_data, Val{col2}(), row_idx)
        output[pos2] = Float64(val2)
    end
    return nothing
end

function execute_operation!(data::ContinuousData{3, Cols}, op::ContinuousOp{3, Cols}, 
                           output, input_data, row_idx) where {Cols}
    @inbounds begin
        # First variable
        col1 = data.columns[1]
        pos1 = data.positions[1]
        val1 = get_data_value_type_stable(input_data, Val{col1}(), row_idx)
        output[pos1] = Float64(val1)
        
        # Second variable  
        col2 = data.columns[2]
        pos2 = data.positions[2]
        val2 = get_data_value_type_stable(input_data, Val{col2}(), row_idx)
        output[pos2] = Float64(val2)
        
        # Third variable
        col3 = data.columns[3]
        pos3 = data.positions[3]
        val3 = get_data_value_type_stable(input_data, Val{col3}(), row_idx)
        output[pos3] = Float64(val3)
    end
    return nothing
end

# General fallback for larger N
function execute_operation!(data::ContinuousData{N, Cols}, op::ContinuousOp{N, Cols}, 
                           output, input_data, row_idx) where {N, Cols}
    
    @inbounds for i in 1:N
        col = data.columns[i]
        pos = data.positions[i]
        val = get_data_value_type_stable(input_data, Val{col}(), row_idx)
        output[pos] = Float64(val)
    end
    
    return nothing
end

"""
    execute_operation!(data::ConstantData{N}, op::ConstantOp{N}, 
                      output, input_data, row_idx) where N

Execute constant operations.
"""
function execute_operation!(data::ConstantData{N}, op::ConstantOp{N}, 
                           output, input_data, row_idx) where N
    
    @inbounds for i in 1:N
        pos = data.positions[i]
        val = data.values[i]
        output[pos] = val
    end
    
    return nothing
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

"""
    Base.length(sf::SpecializedFormula) -> Int

Get the output width of a specialized formula.
"""
Base.length(sf::SpecializedFormula) = sf.output_width
