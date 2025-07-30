# step2_categorical_support.jl
# Add categorical variable support to the specialized formula system

###############################################################################
# CATEGORICAL DATA TYPES (SIMPLIFIED - NO TYPE PARAMETER)
###############################################################################

"""
    CategoricalData

Pre-computed data for categorical variables (simplified, no type parameter).
"""
struct CategoricalData
    contrast_matrix::Matrix{Float64}  # Pre-computed contrast matrix
    level_codes::Vector{Int}          # Pre-extracted level codes for all rows
    positions::Vector{Int}            # Output positions for contrast columns
    n_levels::Int                     # Number of categorical levels
    n_contrasts::Int                  # Number of contrast columns
    
    function CategoricalData(contrast_matrix::Matrix{Float64}, 
                            level_codes::Vector{Int}, 
                            positions::Vector{Int}, 
                            n_levels::Int)
        n_contrasts = size(contrast_matrix, 2)
        @assert length(positions) == n_contrasts "Positions length must match contrast columns"
        new(contrast_matrix, level_codes, positions, n_levels, n_contrasts)
    end
end

"""
    CategoricalOp

Compile-time encoding of categorical variable operations (simplified).
"""
struct CategoricalOp end

###############################################################################
# ENHANCED FORMULA DATA TYPES
###############################################################################

"""
    EnhancedFormulaData{ConstData, ContData, CatData}

Combined data for formulas with constants, continuous, and categorical variables.
CatData is now Vector{CategoricalData} for homogeneous storage.
"""
struct EnhancedFormulaData{ConstData, ContData, CatData}
    constants::ConstData
    continuous::ContData
    categorical::CatData  # This will be Vector{CategoricalData}
end

"""
    EnhancedFormulaOp{ConstOp, ContOp, CatOp}

Combined operation encoding for enhanced formulas.
"""
struct EnhancedFormulaOp{ConstOp, ContOp, CatOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp  # This will be CategoricalOp
end

###############################################################################
# ENHANCED ANALYSIS FUNCTIONS
###############################################################################

"""
    analyze_categorical_operations(evaluator::CombinedEvaluator) -> (Vector{CategoricalData}, CategoricalOp)

Extract categorical data from a CombinedEvaluator's categorical evaluators.
Returns homogeneous vector for allocation-free iteration.
"""
function analyze_categorical_operations(evaluator::CombinedEvaluator)
    categorical_evaluators = evaluator.categorical_evaluators
    n_cats = length(categorical_evaluators)
    
    if n_cats == 0
        # No categorical operations - return empty vector
        return CategoricalData[], CategoricalOp()
    end
    
    # Create vector of homogeneous CategoricalData (no type parameters)
    categorical_data = Vector{CategoricalData}(undef, n_cats)
    
    for i in 1:n_cats
        cat_eval = categorical_evaluators[i]
        
        categorical_data[i] = CategoricalData(
            cat_eval.contrast_matrix,
            cat_eval.level_codes,
            collect(cat_eval.positions),  # Convert to Vector{Int}
            cat_eval.n_levels
        )
    end
    
    return categorical_data, CategoricalOp()
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
    execute_operation!(data::CategoricalData, op::CategoricalOp, 
                      output, input_data, row_idx)

Execute categorical variable operations with zero allocations.
"""
function execute_operation!(data::CategoricalData, op::CategoricalOp, 
                           output, input_data, row_idx)
    
    # Get level for this row (pre-extracted during compilation)
    level = data.level_codes[row_idx]
    
    # Clamp to valid range (safety check)
    level = clamp(level, 1, data.n_levels)
    
    # Direct contrast matrix lookup and assignment
    @inbounds for i in 1:data.n_contrasts
        pos = data.positions[i]
        output[pos] = data.contrast_matrix[level, i]
    end
    
    return nothing
end

"""
    execute_categorical_operations!(categorical_data::Vector{CategoricalData}, output, input_data, row_idx)

Execute multiple categorical variables with zero allocations.
"""
function execute_categorical_operations!(categorical_data::Vector{CategoricalData}, output, input_data, row_idx)
    # Handle empty case
    if isempty(categorical_data)
        return nothing
    end
    
    # Homogeneous vector iteration - no allocations!
    @inbounds for cat_data in categorical_data
        level = cat_data.level_codes[row_idx]
        level = clamp(level, 1, cat_data.n_levels)
        
        for i in 1:cat_data.n_contrasts
            pos = cat_data.positions[i]
            output[pos] = cat_data.contrast_matrix[level, i]
        end
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
        
        if !isempty(sf.data.categorical)
            println("  Categorical variables:")
            for (i, cat_data) in enumerate(sf.data.categorical)
                n_levels = cat_data.n_levels
                n_contrasts = cat_data.n_contrasts
                println("    Categorical $i: $n_levels levels, $n_contrasts contrasts")
            end
        else
            println("  Categorical variables: none")
        end
    end
end
