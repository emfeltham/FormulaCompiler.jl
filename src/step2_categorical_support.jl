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
    
    function CategoricalData(
        contrast_matrix::Matrix{Float64}, 
        level_codes::Vector{Int}, 
        positions::Vector{Int}, 
        n_levels::Int
    )
        n_contrasts = size(contrast_matrix, 2)
        @assert length(positions) == n_contrasts "Positions length must match contrast columns"
        new(contrast_matrix, level_codes, positions, n_levels, n_contrasts)
    end
end

###############################################################################
# Fully Specialized Categorical Data Types
###############################################################################

"""
    SpecializedCategoricalData{N, Positions}

Fully compile-time specialized categorical data with tuple-based positions.
N = number of contrast columns, Positions = NTuple{N, Int} of output positions.
"""
struct SpecializedCategoricalData{N, Positions}
    contrast_matrix::Matrix{Float64}
    level_codes::Vector{Int}
    positions::Positions                  # NTuple{N, Int}
    n_levels::Int
    n_contrasts::Int
    
    function SpecializedCategoricalData(
        contrast_matrix::Matrix{Float64}, 
        level_codes::Vector{Int}, 
        positions::NTuple{N, Int}, 
        n_levels::Int
    ) where N
        n_contrasts = size(contrast_matrix, 2)
        @assert N == n_contrasts "Position tuple length must match contrast columns"
        new{N, typeof(positions)}(contrast_matrix, level_codes, positions, n_levels, n_contrasts)
    end
end

"""
    CategoricalOp{N}

Compile-time encoding of categorical operations with known count.
"""
struct CategoricalOp{N}
    function CategoricalOp(n::Int) 
        new{n}()
    end
end

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
Fully specialized analysis that returns compile-time tuples.
"""
function analyze_categorical_operations(evaluator::CombinedEvaluator)
    categorical_evaluators = evaluator.categorical_evaluators
    n_cats = length(categorical_evaluators)
    
    if n_cats == 0
        # No categorical operations - return empty tuple
        return (), CategoricalOp(0)
    end
    
    # Create tuple of specialized categorical data using ntuple
    categorical_data = ntuple(n_cats) do i
        cat_eval = categorical_evaluators[i]
        
        # Convert positions vector to compile-time tuple
        position_tuple = ntuple(length(cat_eval.positions)) do j
            cat_eval.positions[j]
        end
        
        SpecializedCategoricalData(
            cat_eval.contrast_matrix,
            cat_eval.level_codes,
            position_tuple,  # Now a compile-time tuple!
            cat_eval.n_levels
        )
    end
    
    return categorical_data, CategoricalOp(n_cats)
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

###############################################################################
# RECURSIVE TUPLE PROCESSING FOR CATEGORICAL EXECUTION
###############################################################################

"""
    execute_categorical_recursive!(categorical_data::Tuple{}, output, input_data, row_idx)

Base case: empty tuple - nothing to process.
"""
function execute_categorical_recursive!(
    categorical_data::Tuple{}, 
    output, 
    input_data, 
    row_idx
)
    # Base case: no categoricals to process
    return nothing
end

"""
    execute_categorical_recursive!(categorical_data::Tuple, output, input_data, row_idx)

Recursive case: process first categorical, then recursively process the rest.
Fixed to avoid TypeVar iteration issues during precompilation.
"""
function execute_categorical_recursive!(
    categorical_data::Tuple, 
    output, 
    input_data, 
    row_idx
)
    # Handle empty tuple (should be caught by specialized method above)
    if length(categorical_data) == 0
        return nothing
    end
    
    # Process the first categorical variable
    cat_data = categorical_data[1]  # First element
    
    # Get level for this row
    level = cat_data.level_codes[row_idx]
    level = clamp(level, 1, cat_data.n_levels)
    
    # Execute this categorical using existing specialized function
    execute_single_categorical!(cat_data, level, output)
    
    # Recursively process the remaining categoricals
    if length(categorical_data) > 1
        remaining_data = Base.tail(categorical_data)  # Get tail
        execute_categorical_recursive!(remaining_data, output, input_data, row_idx)
    end
    
    return nothing
end

###############################################################################
# Main categorical execution function
###############################################################################

"""
Use recursive tuple processing.
"""
function execute_categorical_operations!(
    categorical_data::Tuple, 
    output, 
    input_data, 
    row_idx
)
    # Use recursive processing
    execute_categorical_recursive!(categorical_data, output, input_data, row_idx)
    return nothing
end

"""
    execute_single_categorical!(cat_data, level, output)

Recursive approach calls this for each categorical variable.
Optimized handling when level_codes come from OverrideVector.
"""
function execute_single_categorical!(
    cat_data::SpecializedCategoricalData{N, Positions}, 
    level::Int, 
    output
) where {N, Positions}
    
    # Level is already clamped by the caller
    # Just do the matrix lookup and assignment
    @inbounds for i in 1:N  # N is compile-time constant!
        pos = cat_data.positions[i]  # Position known at compile time
        output[pos] = cat_data.contrast_matrix[level, i]
    end
    
    return nothing
end
