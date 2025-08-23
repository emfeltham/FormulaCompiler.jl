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

FIXED: Remove pre-computed level_codes, extract them dynamically during execution
while maintaining zero allocations.
"""
struct SpecializedCategoricalData{N, Positions}
    contrast_matrix::Matrix{Float64}
    positions::Positions                  # NTuple{N, Int}
    n_levels::Int
    n_contrasts::Int
    column::Symbol                        # ADDED: Need to know which column to read from
    
    function SpecializedCategoricalData(
        contrast_matrix::Matrix{Float64}, 
        positions::NTuple{N, Int}, 
        n_levels::Int,
        column::Symbol
    ) where N
        n_contrasts = size(contrast_matrix, 2)
        @assert N == n_contrasts "Position tuple length must match contrast columns"
        new{N, typeof(positions)}(contrast_matrix, positions, n_levels, n_contrasts, column)
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
    analyze_categorical_operations(evaluator::CombinedEvaluator) -> (Tuple, CategoricalOp)

FIXED: Don't pre-extract level codes during compilation.
"""
function analyze_categorical_operations(evaluator::CombinedEvaluator)
    categorical_evaluators = evaluator.categorical_evaluators
    n_cats = length(categorical_evaluators)
    
    if n_cats == 0
        return (), CategoricalOp(0)
    end
    
    # Create tuple of specialized categorical data using ntuple
    categorical_data = ntuple(n_cats) do i
        cat_eval = categorical_evaluators[i]
        
        # Convert positions vector to compile-time tuple
        position_tuple = ntuple(length(cat_eval.positions)) do j
            cat_eval.positions[j]
        end
        
        # FIXED: Pass column symbol for dynamic lookup, not pre-computed level codes
        SpecializedCategoricalData(
            cat_eval.contrast_matrix,
            position_tuple,
            cat_eval.n_levels,
            cat_eval.column  # Column name for dynamic lookup
        )
    end
    
    return categorical_data, CategoricalOp(n_cats)
end

"""
    extract_level_code_zero_alloc(column_data, row_idx::Int) -> Int

Extract level code with zero allocations using type-stable dispatch.
"""
@inline function extract_level_code_zero_alloc(column_data::CategoricalVector, row_idx::Int)
    return Int(levelcode(column_data[row_idx]))
end

@inline function extract_level_code_zero_alloc(column_data::OverrideVector{CategoricalValue{T,R}}, row_idx::Int) where {T,R}
    # For OverrideVector, all rows have the same value - extract once, no allocation
    return Int(levelcode(column_data.override_value))
end

@inline function extract_level_code_zero_alloc(column_data, row_idx::Int)
    # Fallback for other types
    error("Cannot extract level code from $(typeof(column_data))")
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

FIXED: Extract level codes dynamically with zero allocations.
This REPLACES the existing function completely.
"""
function execute_categorical_recursive!(
    categorical_data::Tuple, 
    output, 
    input_data, 
    row_idx
)
    # Base case: empty tuple
    if length(categorical_data) == 0
        return nothing
    end
    
    # Process the first categorical variable
    cat_data = categorical_data[1]
    
    # FIXED: Extract level dynamically with zero allocations
    column_data = getproperty(input_data, cat_data.column)  # Zero-allocation property access
    level = extract_level_code_zero_alloc(column_data, row_idx)
    
    # Execute this categorical using existing zero-allocation function
    execute_single_categorical!(cat_data, level, output)
    
    # Recursively process remaining categoricals
    if length(categorical_data) > 1
        remaining_data = Base.tail(categorical_data)
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

    # println("DEBUG: level=$level, contrast_matrix size=$(size(cat_data.contrast_matrix))")
    # println("DEBUG: contrast row for level $level: $(cat_data.contrast_matrix[level, :])")
    
    # Level is already clamped by the caller
    # Just do the matrix lookup and assignment
    @inbounds for i in 1:N  # N is compile-time constant!
        pos = cat_data.positions[i]  # Position known at compile time
        output[pos] = cat_data.contrast_matrix[level, i]
    end
    
    return nothing
end

###############################################################################
# 4. ZERO-ALLOCATION VERIFICATION TEST
###############################################################################

"""
    verify_categorical_fix()

Test function to verify the fix works correctly with zero allocations.
Run this after implementing the changes.
"""
function verify_categorical_fix()
    println("Testing categorical scenario fix...")
    
    # Create test data
    n = 100
    df = DataFrame(
        x = randn(n),
        group = categorical(rand(["A", "B", "C"], n))
    )
    df.y = randn(n)
    
    model = lm(@formula(y ~ x + group), df)
    data_nt = Tables.columntable(df)
    
    # Create scenarios
    scenario_A = create_scenario("A", data_nt; group = "A")
    scenario_B = create_scenario("B", data_nt; group = "B")
    scenario_C = create_scenario("C", data_nt; group = "C")
    
    # Compile formula
    compiled = compile_formula(model, scenario_A.data)
    buffer = Vector{Float64}(undef, length(compiled))
    
    # Test zero allocations
    println("Testing zero allocations...")
    alloc_test = @allocated begin
        for i in 1:100
            compiled(buffer, scenario_A.data, 1)
            compiled(buffer, scenario_B.data, 1)  
            compiled(buffer, scenario_C.data, 1)
        end
    end
    println("Allocations for 300 evaluations: $alloc_test bytes")
    
    # Test correctness - results should be different
    compiled(buffer, scenario_A.data, 1)
    result_A = copy(buffer)
    
    compiled(buffer, scenario_B.data, 1)
    result_B = copy(buffer)
    
    compiled(buffer, scenario_C.data, 1) 
    result_C = copy(buffer)
    
    println("Results:")
    println("  Scenario A: $result_A")
    println("  Scenario B: $result_B") 
    println("  Scenario C: $result_C")
    
    println("Results are different:")
    println("  A ≠ B: $(result_A != result_B)")
    println("  B ≠ C: $(result_B != result_C)")
    println("  A ≠ C: $(result_A != result_C)")
    
    # Test marginal effects
    println("\nTesting marginal effects...")
    result = margins(model, df, :group)
    println("Marginal effects result:")
    show(result)
    
    return (alloc_test, result_A, result_B, result_C, result)
end
