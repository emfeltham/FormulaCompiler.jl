# step2_categorical_support.jl
# Add categorical variable support to the specialized formula system

###############################################################################
# CATEGORICAL DATA TYPES
###############################################################################

"""
    CategoricalData{N}

Pre-computed data for categorical variables with N contrast columns.
"""
struct CategoricalData{N}
    contrast_matrix::Matrix{Float64}  # Pre-computed contrast matrix
    level_codes::Vector{Int}          # Pre-extracted level codes for all rows
    positions::NTuple{N, Int}         # Output positions for contrast columns
    n_levels::Int                     # Number of categorical levels
    
    # Inner constructor with explicit N parameter
    function CategoricalData{N}(contrast_matrix::Matrix{Float64}, 
                               level_codes::Vector{Int}, 
                               positions::NTuple{N, Int}, 
                               n_levels::Int) where N
        @assert size(contrast_matrix, 2) == N "Contrast matrix columns must match N"
        new{N}(contrast_matrix, level_codes, positions, n_levels)
    end
end

"""
    CategoricalOp{N}

Compile-time encoding of categorical variable operations with N contrast columns.
"""
struct CategoricalOp{N}
    function CategoricalOp(::CategoricalData{N}) where N
        new{N}()
    end
end

###############################################################################
# ENHANCED FORMULA DATA TYPES
###############################################################################

"""
    EnhancedFormulaData{ConstData, ContData, CatData}

Combined data for formulas with constants, continuous, and categorical variables.
"""
struct EnhancedFormulaData{ConstData, ContData, CatData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
end

"""
    EnhancedFormulaOp{ConstOp, ContOp, CatOp}

Combined operation encoding for enhanced formulas.
"""
struct EnhancedFormulaOp{ConstOp, ContOp, CatOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp
end

###############################################################################
# ENHANCED ANALYSIS FUNCTIONS
###############################################################################

"""
    analyze_categorical_operations(evaluator::CombinedEvaluator) -> (CategoricalData, CategoricalOp)

Extract categorical data from a CombinedEvaluator's categorical evaluators.
"""
function analyze_categorical_operations(evaluator::CombinedEvaluator)
    categorical_evaluators = evaluator.categorical_evaluators
    n_cats = length(categorical_evaluators)
    
    if n_cats == 0
        # No categorical operations - return empty tuple
        return (), ()
    elseif n_cats == 1
        # Single categorical variable
        cat_eval = categorical_evaluators[1]
        N = length(cat_eval.positions)
        
        categorical_data = CategoricalData{N}(
            cat_eval.contrast_matrix,
            cat_eval.level_codes,
            Tuple(cat_eval.positions),
            cat_eval.n_levels
        )
        categorical_op = CategoricalOp(categorical_data)
        
        return (categorical_data,), (categorical_op,)
    else
        # Multiple categorical variables - create tuple of data
        categorical_data_tuple = ntuple(n_cats) do i
            cat_eval = categorical_evaluators[i]
            N = length(cat_eval.positions)
            
            CategoricalData{N}(
                cat_eval.contrast_matrix,
                cat_eval.level_codes,
                Tuple(cat_eval.positions),
                cat_eval.n_levels
            )
        end
        
        categorical_op_tuple = ntuple(n_cats) do i
            CategoricalOp(categorical_data_tuple[i])
        end
        
        return categorical_data_tuple, categorical_op_tuple
    end
end

"""
    analyze_evaluator_enhanced(evaluator::AbstractEvaluator) -> (DataTuple, OpTuple)

Enhanced analysis for constants, continuous, and categorical variables.
"""
function analyze_evaluator_enhanced(evaluator::AbstractEvaluator)
    if evaluator isa CombinedEvaluator
        # Check that this only has simple operation types (no functions/interactions yet)
        has_complex_operations = (
            !isempty(evaluator.function_evaluators) ||
            !isempty(evaluator.interaction_evaluators)
        )
        
        if has_complex_operations
            error("Step 2 only supports constants, continuous, and categorical variables. Found functions/interactions.")
        end
        
        # Analyze all three operation types
        constant_data, constant_op = analyze_constant_operations(evaluator)
        continuous_data, continuous_op = analyze_continuous_operations(evaluator)
        categorical_data, categorical_op = analyze_categorical_operations(evaluator)
        
        # Combine into enhanced formula data
        formula_data = EnhancedFormulaData(constant_data, continuous_data, categorical_data)
        formula_op = EnhancedFormulaOp(constant_op, continuous_op, categorical_op)
        
        return formula_data, formula_op
        
    else
        error("Step 2 only supports CombinedEvaluator with constants, continuous, and categorical operations")
    end
end

###############################################################################
# CATEGORICAL EXECUTION FUNCTIONS
###############################################################################

"""
    execute_operation!(data::CategoricalData{N}, op::CategoricalOp{N}, 
                      output, input_data, row_idx) where N

Execute categorical variable operations with zero allocations.
"""
function execute_operation!(data::CategoricalData{N}, op::CategoricalOp{N}, 
                           output, input_data, row_idx) where N
    
    # Get level for this row (pre-extracted during compilation)
    level = data.level_codes[row_idx]
    
    # Clamp to valid range (safety check)
    level = clamp(level, 1, data.n_levels)
    
    # Direct contrast matrix lookup and assignment
    @inbounds for i in 1:N
        pos = data.positions[i]
        output[pos] = data.contrast_matrix[level, i]
    end
    
    return nothing
end

"""
    execute_categorical_operations!(categorical_data::Tuple, output, input_data, row_idx)

Execute multiple categorical variables.
"""
function execute_categorical_operations!(categorical_data::Tuple, output, input_data, row_idx)
    for cat_data in categorical_data
        execute_categorical_operation_single!(cat_data, output, input_data, row_idx)
    end
    return nothing
end

function execute_categorical_operations!(categorical_data::Tuple{}, output, input_data, row_idx)
    # No categorical variables - do nothing
    return nothing
end

"""
    execute_categorical_operation_single!(data::CategoricalData{N}, output, input_data, row_idx) where N

Execute a single categorical variable operation.
"""
function execute_categorical_operation_single!(data::CategoricalData{N}, output, input_data, row_idx) where N
    level = data.level_codes[row_idx]
    level = clamp(level, 1, data.n_levels)
    
    @inbounds for i in 1:N
        pos = data.positions[i]
        output[pos] = data.contrast_matrix[level, i]
    end
    
    return nothing
end

"""
    execute_operation!(data::EnhancedFormulaData{ConstData, ContData, CatData}, 
                      op::EnhancedFormulaOp{ConstOp, ContOp, CatOp}, 
                      output, input_data, row_idx) where {ConstData, ContData, CatData, ConstOp, ContOp, CatOp}

Execute enhanced formulas with constants, continuous, and categorical variables.
"""
function execute_operation!(data::EnhancedFormulaData{ConstData, ContData, CatData}, 
                           op::EnhancedFormulaOp{ConstOp, ContOp, CatOp}, 
                           output, input_data, row_idx) where {ConstData, ContData, CatData, ConstOp, ContOp, CatOp}
    
    # Execute constants
    execute_operation!(data.constants, op.constants, output, input_data, row_idx)
    
    # Execute continuous variables
    execute_operation!(data.continuous, op.continuous, output, input_data, row_idx)
    
    # Execute categorical variables
    execute_categorical_operations!(data.categorical, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# ENHANCED COMPILATION FUNCTIONS
###############################################################################

"""
    create_specialized_formula_enhanced(compiled_formula::CompiledFormula) -> SpecializedFormula

Convert a CompiledFormula to a SpecializedFormula (Step 2: includes categorical).
"""
function create_specialized_formula_enhanced(compiled_formula::CompiledFormula)
    # Analyze the evaluator tree with enhanced support
    data_tuple, op_tuple = analyze_evaluator_enhanced(compiled_formula.root_evaluator)
    
    # Create specialized formula
    return SpecializedFormula{typeof(data_tuple), typeof(op_tuple)}(
        data_tuple,
        op_tuple,
        compiled_formula.output_width
    )
end

"""
    compile_formula_specialized_enhanced(model, data::NamedTuple) -> SpecializedFormula

Direct compilation to specialized formula (Step 2: includes categorical).
"""
function compile_formula_specialized_enhanced(model, data::NamedTuple)
    # Use existing compilation logic to build evaluator tree
    compiled = compile_formula(model, data)
    
    # Convert to enhanced specialized form
    return create_specialized_formula_enhanced(compiled)
end

###############################################################################
# ENHANCED UTILITY FUNCTIONS
###############################################################################

"""
    show_enhanced_specialized_info(sf::SpecializedFormula)

Display detailed information about an enhanced specialized formula.
"""
function show_enhanced_specialized_info(sf::SpecializedFormula{D, O}) where {D, O}
    println("Enhanced SpecializedFormula Information:")
    println("  Data type: $D")
    println("  Operation type: $O") 
    println("  Output width: $(sf.output_width)")
    
    if sf.data isa EnhancedFormulaData
        println("  Constants: $(sf.data.constants.values)")
        println("  Continuous variables: $(sf.data.continuous.columns)")
        
        if sf.data.categorical isa Tuple && !isempty(sf.data.categorical)
            println("  Categorical variables:")
            for (i, cat_data) in enumerate(sf.data.categorical)
                n_levels = cat_data.n_levels
                n_contrasts = length(cat_data.positions)
                println("    Categorical $i: $n_levels levels, $n_contrasts contrasts")
            end
        else
            println("  Categorical variables: none")
        end
    end
end