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
    
    # Extract columns and positions
    columns = ntuple(n_ops) do i
        continuous_ops[i].column
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
function execute_operation!(data::ContinuousData{N, Cols}, op::ContinuousOp{N, Cols}, 
                           output, input_data, row_idx) where {N, Cols}
    
    @inbounds for i in 1:N
        col = data.columns[i]
        pos = data.positions[i]
        val = get_data_value_specialized(input_data, col, row_idx)
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
