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
# OVERWRITE: Main categorical execution function
###############################################################################

"""
Overwrite the main categorical execution to use recursive tuple processing.
This replaces the loop-based approach with allocation-free recursion.
"""
function execute_categorical_operations!(
    categorical_data::Tuple, 
    output, 
    input_data, 
    row_idx
)
    # Use recursive processing instead of loops
    execute_categorical_recursive!(categorical_data, output, input_data, row_idx)
    return nothing
end

###############################################################################
# KEEP: Existing single categorical execution (unchanged)
###############################################################################

"""
Keep the existing single categorical execution function.
The recursive approach calls this for each categorical variable.
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
# DEBUGGING AND VALIDATION
###############################################################################

"""
    trace_recursive_execution(categorical_data::Tuple, description="")

Debug function to trace how the recursive execution unfolds.
"""
function trace_recursive_execution(categorical_data::Tuple, description="")
    println("Tracing recursive execution: $description")
    println("  Input tuple type: $(typeof(categorical_data))")
    println("  Tuple length: $(length(categorical_data))")
    
    if length(categorical_data) == 0
        println("  → Base case: empty tuple")
    else
        println("  → Recursive case:")
        println("    First element type: $(typeof(categorical_data[1]))")
        if length(categorical_data) > 1
            println("    Remaining elements: $(typeof(Base.tail(categorical_data)))")
        else
            println("    Remaining elements: (none - will hit base case)")
        end
    end
    
    return nothing
end

"""
    benchmark_recursive_categorical(formula, df, data; n_iterations=1000)

Benchmark the recursive categorical execution approach.
"""
function benchmark_recursive_categorical(formula, df, data; n_iterations=1000)
    println("Benchmarking recursive categorical execution...")
    println("Formula: $formula")
    
    # Compile the formula
    model = fit(LinearModel, formula, df)
    compiled = compile_formula_specialized(model, data)
    output = Vector{Float64}(undef, length(compiled))
    
    println("Compiled type: $(typeof(compiled.data.categorical))")
    
    # Trace the recursion structure
    trace_recursive_execution(compiled.data.categorical, "Compiled formula")
    
    # Warmup
    for _ in 1:20
        compiled(output, data, 1)
    end
    
    # Benchmark allocation
    alloc = @allocated begin
        for i in 1:n_iterations
            row_idx = ((i - 1) % length(data.x)) + 1
            compiled(output, data, row_idx)
        end
    end
    
    avg_alloc = alloc / n_iterations
    
    println("Performance results:")
    println("  Iterations: $n_iterations")
    println("  Total allocations: $alloc bytes")
    println("  Average per call: $avg_alloc bytes")
    
    if avg_alloc == 0
        println("  🎯 PERFECT: Zero allocations achieved!")
    elseif avg_alloc <= 32
        println("  ✅ EXCELLENT: ≤32 bytes per call")
    elseif avg_alloc <= 64
        println("  ✅ GOOD: ≤64 bytes per call")
    else
        println("  ⚠️  NEEDS WORK: >64 bytes per call")
    end
    
    # Test correctness
    test_output1 = Vector{Float64}(undef, length(compiled))
    test_output2 = Vector{Float64}(undef, length(compiled))
    
    compiled(test_output1, data, 1)
    compiled(test_output2, data, 5)
    
    println("  Sample outputs look reasonable: $(all(isfinite, test_output1) && all(isfinite, test_output2))")
    
    return avg_alloc
end

"""
    test_recursive_approach()

Test the recursive approach on various categorical configurations.
"""
function test_recursive_approach()
    println("="^60)
    println("TESTING RECURSIVE CATEGORICAL EXECUTION")
    println("="^60)
    
    # Create test data
    n = 100
    df = DataFrame(
        x = randn(n),
        group3 = categorical(rand(["A", "B", "C"], n)),           
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      
        binary = categorical(rand(["Yes", "No"], n)),             
        response = randn(n)
    )
    data = Tables.columntable(df)
    
    # Test cases
    test_cases = [
        (@formula(response ~ 1), "No categoricals"),
        (@formula(response ~ group3), "1 categorical"),
        (@formula(response ~ group3 + group4), "2 categoricals"),
        (@formula(response ~ group3 + group4 + binary), "3 categoricals"),
        (@formula(response ~ x + group3 + group4 + binary), "3 categoricals + continuous"),
    ]
    
    results = []
    for (formula, description) in test_cases
        println("\n" * "-"^40)
        result = benchmark_recursive_categorical(formula, df, data, n_iterations=100)
        push!(results, (description, result))
    end
    
    # Summary
    println("\n" * "="^40)
    println("RECURSIVE APPROACH SUMMARY")
    println("="^40)
    
    for (description, result) in results
        status = result == 0 ? "🎯" : result <= 32 ? "✅" : "⚠️"
        println("$status $description: $result bytes per call")
    end
    
    # Check if the 3-categorical problem is solved
    three_cat_result = results[findfirst(r -> r[1] == "3 categoricals", results)][2]
    if three_cat_result == 0
        println("\n🎉 SUCCESS: 3-categorical allocation problem solved!")
    elseif three_cat_result < 100
        println("\n✅ MAJOR IMPROVEMENT: 3-categorical allocations greatly reduced")
    else
        println("\n⚠️  PARTIAL: 3-categorical allocations reduced but not eliminated")
    end
    
    return results
end
