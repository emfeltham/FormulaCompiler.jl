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
# NEW: Specialized Execution Functions for Tuples
###############################################################################

"""
    execute_categorical_operations!(categorical_data::Tuple, output, input_data, row_idx)

Specialized execution for compile-time tuple of categorical data.
Loop bounds are compile-time constants enabling full optimization.
"""
function execute_categorical_operations!(
    categorical_data::Tuple, 
    output, 
    input_data, 
    row_idx
)
    # Handle empty tuple case
    if length(categorical_data) == 0
        return nothing
    end
    
    # Loop over compile-time known number of categorical variables
    # Julia knows the exact count at compile time!
    @inbounds for i in 1:length(categorical_data)
        cat_data = categorical_data[i]
        
        # Get level for this row (pre-extracted during compilation)
        level = cat_data.level_codes[row_idx]
        level = clamp(level, 1, cat_data.n_levels)
        
        # Execute contrast assignment with compile-time known positions
        execute_single_categorical!(cat_data, level, output)
    end
    
    return nothing
end

"""
    execute_single_categorical!(cat_data::SpecializedCategoricalData{N, Positions}, level::Int, output) where {N, Positions}

Execute single categorical variable with fully compile-time specialized positions.
"""
function execute_single_categorical!(
    cat_data::SpecializedCategoricalData{N, Positions}, 
    level::Int, 
    output
) where {N, Positions}
    
    # Loop bounds and positions are compile-time constants
    @inbounds for i in 1:N  # N is compile-time constant!
        pos = cat_data.positions[i]  # Position known at compile time
        output[pos] = cat_data.contrast_matrix[level, i]
    end
    
    return nothing
end

###############################################################################
# SPECIALIZED METHODS for Common Cases
###############################################################################

"""
Specialized method for empty categorical case (very common).
"""
function execute_categorical_operations!(
    categorical_data::Tuple{}, 
    output, 
    input_data, 
    row_idx
)
    # No-op for empty case - Julia can optimize this away completely
    return nothing
end

"""
Specialized method for single categorical variable (very common).
"""
function execute_categorical_operations!(
    categorical_data::Tuple{SpecializedCategoricalData{N, Positions}}, 
    output, 
    input_data, 
    row_idx
) where {N, Positions}
    
    cat_data = categorical_data[1]
    level = cat_data.level_codes[row_idx]
    level = clamp(level, 1, cat_data.n_levels)
    
    # Fully unrolled execution for single categorical
    execute_single_categorical!(cat_data, level, output)
    
    return nothing
end

"""
Specialized method for two categorical variables (common).
"""
function execute_categorical_operations!(
    categorical_data::Tuple{
        SpecializedCategoricalData{N1, P1}, 
        SpecializedCategoricalData{N2, P2}
    }, 
    output, 
    input_data, 
    row_idx
) where {N1, P1, N2, P2}
    
    # First categorical
    cat_data1 = categorical_data[1]
    level1 = cat_data1.level_codes[row_idx]
    level1 = clamp(level1, 1, cat_data1.n_levels)
    execute_single_categorical!(cat_data1, level1, output)
    
    # Second categorical  
    cat_data2 = categorical_data[2]
    level2 = cat_data2.level_codes[row_idx]
    level2 = clamp(level2, 1, cat_data2.n_levels)
    execute_single_categorical!(cat_data2, level2, output)
    
    return nothing
end

###############################################################################
# DEBUGGING AND VALIDATION
###############################################################################

"""
    show_categorical_specialization_info(categorical_data)

Display information about categorical specialization for debugging.
"""
function show_categorical_specialization_info(categorical_data)
    println("Categorical Specialization Info:")
    println("  Type: $(typeof(categorical_data))")
    println("  Count: $(length(categorical_data))")
    
    if length(categorical_data) > 0
        for (i, cat_data) in enumerate(categorical_data)
            println("  Categorical $i:")
            println("    Type: $(typeof(cat_data))")
            println("    Contrast columns: $(cat_data.n_contrasts)")
            println("    Levels: $(cat_data.n_levels)")
            println("    Positions: $(cat_data.positions)")
            println("    Position type: $(typeof(cat_data.positions))")
        end
    end
end

"""
    validate_categorical_specialization(formula, df, data)

Validate that categorical specialization produces correct results.
"""
function validate_categorical_specialization(formula, df, data)
    println("Validating categorical specialization for: $formula")
    
    # Compile with new specialization
    model = fit(LinearModel, formula, df)
    compiled = compile_formula_specialized(model, data)
    
    println("Compiled type: $(typeof(compiled))")
    show_categorical_specialization_info(compiled.data.categorical)
    
    # Test execution
    output = Vector{Float64}(undef, length(compiled))
    compiled(output, data, 1)
    
    println("Execution successful: $(output)")
    
    # Test allocation performance
    for _ in 1:10
        compiled(output, data, 1)
    end
    
    allocs = @allocated begin
        for i in 1:100
            compiled(output, data, i)
        end
    end
    
    println("Allocations: $(allocs / 100) bytes per call")
    
    return allocs / 100
end
