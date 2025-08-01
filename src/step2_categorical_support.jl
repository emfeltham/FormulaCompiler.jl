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

###############################################################################
# CATEGORICAL EXECUTION FUNCTIONS
###############################################################################

"""
    execute_operation!(data::CategoricalData, op::CategoricalOp, 
                      output, input_data, row_idx)

Execute categorical variable operations.
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

Execute multiple categorical variables.
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
