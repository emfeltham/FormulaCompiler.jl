# Updated modelrow!.jl functions for self-contained evaluator system

###############################################################################
# CACHE VARIABLE
###############################################################################

"""
Cache for compiled formulas - using Tuple key for model+data structure.
"""
const MODEL_IDENTITY_CACHE = Dict{Any, CompiledFormula}()

###############################################################################
# UPDATED CORE MODELROW! FUNCTION
###############################################################################

"""
    modelrow!(row_vec, compiled_formula, data, row_idx)

UPDATED: Zero-allocation model row evaluation using self-contained evaluators.
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    compiled::CompiledFormula, 
    data, 
    row_idx::Int
)
    # bounds check
    @assert length(row_vec) >= length(compiled) "Vector too small: need $(length(compiled)), got $(length(row_vec))"
    @assert 1 <= row_idx <= length(first(data)) "Invalid row index: $row_idx (data has $(length(first(data))) rows)"

    # UPDATED: Use self-contained evaluator execution directly
    execute_self_contained!(compiled.root_evaluator, compiled.scratch_space, row_vec, data, row_idx)
    return row_vec
end

###############################################################################
# UPDATED CONVENIENCE FUNCTION
###############################################################################

"""
    modelrow!(row_vec, model, data, row_idx; cache=true)

UPDATED: Convenient modelrow! using self-contained evaluators.
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    model::Union{LinearModel, GeneralizedLinearModel, LinearMixedModel, GeneralizedLinearMixedModel, StatsModels.TableRegressionModel}, 
    data, 
    row_idx::Int; 
    cache::Bool=true
)
    if cache
        # Get cached compiled formula
        compiled = get_or_compile_formula_identity_updated(model, data)
        execute_self_contained!(compiled.root_evaluator, compiled.scratch_space, row_vec, data, row_idx)
    else
        # Always recompile
        compiled = compile_formula(model, data)  # UPDATED: Now requires data parameter
        execute_self_contained!(compiled.root_evaluator, compiled.scratch_space, row_vec, data, row_idx)
    end
    return row_vec
end

###############################################################################
# UPDATED MODEL ROW EVALUATOR
###############################################################################

"""
    ModelRowEvaluator

UPDATED: Pre-compiled evaluator using self-contained system.
"""
struct ModelRowEvaluator
    compiled::CompiledFormula
    data::NamedTuple
    row_vec::Vector{Float64}
    
    function ModelRowEvaluator(model, df::DataFrame)
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)  # UPDATED: Now requires data parameter
        row_vec = Vector{Float64}(undef, length(compiled))
        new(compiled, data, row_vec)
    end
end

"""
    (evaluator::ModelRowEvaluator)(row_idx) -> Vector{Float64}

UPDATED: Evaluate model row using self-contained execution.
"""
function (evaluator::ModelRowEvaluator)(row_idx::Int)
    execute_self_contained!(evaluator.compiled.root_evaluator, 
                           evaluator.compiled.scratch_space, 
                           evaluator.row_vec, 
                           evaluator.data, 
                           row_idx)
    return evaluator.row_vec
end

"""
    (evaluator::ModelRowEvaluator)(row_vec, row_idx)

UPDATED: Evaluate model row into provided vector using self-contained execution.
"""
function (evaluator::ModelRowEvaluator)(row_vec::AbstractVector{Float64}, row_idx::Int)
    execute_self_contained!(evaluator.compiled.root_evaluator, 
                           evaluator.compiled.scratch_space, 
                           row_vec, 
                           evaluator.data, 
                           row_idx)
    return row_vec
end

###############################################################################
# UPDATED CACHING SYSTEM
###############################################################################

"""
    get_or_compile_formula_identity_updated(model, data)

UPDATED: Caching system that works with new compile_formula(model, data) signature.
"""
function get_or_compile_formula_identity_updated(model, data)
    # Create cache key from both model and data structure
    cache_key = (model, hash(keys(data)))
    
    if haskey(MODEL_IDENTITY_CACHE, cache_key)
        return MODEL_IDENTITY_CACHE[cache_key]
    else
        compiled = compile_formula(model, data)  # UPDATED: Now requires data
        MODEL_IDENTITY_CACHE[cache_key] = compiled
        return compiled
    end
end

"""
    modelrow_cached!(row_vec, model, data, row_idx)

UPDATED: Zero-allocation cached version using self-contained evaluators.
"""
function modelrow_cached!(row_vec::AbstractVector{Float64}, model, data, row_idx::Int)
    compiled = get_or_compile_formula_identity_updated(model, data)
    execute_self_contained!(compiled.root_evaluator, compiled.scratch_space, row_vec, data, row_idx)
    return row_vec
end

###############################################################################
# UPDATED MULTI-ROW FUNCTIONS
###############################################################################

# """
#     modelrow!(matrix, compiled_formula, data, row_indices)

# UPDATED: Zero-allocation multi-row evaluation using self-contained evaluators.
# """
# function modelrow!(
#     matrix::AbstractMatrix{Float64}, 
#     compiled::CompiledFormula, 
#     data, 
#     row_indices::Vector{Int}
# )
#     @assert size(matrix, 1) >= length(row_indices) "Matrix height insufficient for row_indices"
#     @assert size(matrix, 2) == length(compiled) "Matrix width must match compiled formula width"
    
#     # UPDATED: Use self-contained execution for each row
#     for (i, row_idx) in enumerate(row_indices)
#         row_view = view(matrix, i, :)
#         execute_self_contained!(compiled.root_evaluator, compiled.scratch_space, row_view, data, row_idx)
#     end
    
#     return matrix
# end

# """
#     modelrow!(matrix, model, data, row_indices; cache=true)

# UPDATED: Convenient multi-row modelrow! using self-contained evaluators.
# """
# function modelrow!(
#     matrix::AbstractMatrix{Float64}, 
#     model::Union{LinearModel, GeneralizedLinearModel, LinearMixedModel, GeneralizedLinearMixedModel, StatsModels.TableRegressionModel}, 
#     data, 
#     row_indices::Vector{Int}; 
#     cache::Bool=true
# )
#     if cache
#         compiled = get_or_compile_formula_identity_updated(model, data)
#     else
#         compiled = compile_formula(model, data)  # UPDATED: Now requires data
#     end
    
#     # Delegate to the updated pre-compiled version
#     modelrow!(matrix, compiled, data, row_indices)
#     return matrix
# end

# """
#     modelrow_colmajor!(matrix, compiled_formula, data, row_indices)

# UPDATED: Column-major optimized multi-row evaluation using self-contained evaluators.
# """
# function modelrow_colmajor!(
#     matrix::AbstractMatrix{Float64}, 
#     compiled::CompiledFormula, 
#     data, 
#     row_indices::Vector{Int}
# )
#     @assert size(matrix, 1) >= length(row_indices) "Matrix height insufficient for row_indices"
#     @assert size(matrix, 2) == length(compiled) "Matrix width must match compiled formula width"
    
#     n_rows = length(row_indices)
#     n_cols = length(compiled)
    
#     # UPDATED: Use self-contained execution with temporary row
#     temp_row = Vector{Float64}(undef, n_cols)
    
#     for (i, row_idx) in enumerate(row_indices)
#         # Evaluate into temporary row using self-contained execution
#         execute_self_contained!(compiled.root_evaluator, compiled.scratch_space, temp_row, data, row_idx)
        
#         # Copy to matrix column by column (cache-friendly)
#         for j in 1:n_cols
#             matrix[i, j] = temp_row[j]
#         end
#     end
    
#     return matrix
# end

###############################################################################
# TESTING FUNCTION FOR UPDATED SYSTEM
###############################################################################

"""
    test_updated_modelrow_system()

Test the updated modelrow! system with self-contained evaluators.
"""
function test_updated_modelrow_system()
    println("🧪 TESTING UPDATED MODELROW! SYSTEM")
    println("=" ^ 50)
    
    # Create test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0],
        y = [4.0, 5.0, 6.0],
        group = categorical(["A", "B", "A"])
    )
    data = Tables.columntable(df)
    
    # Test cases with different complexity
    test_cases = [
        (@formula(y ~ x), "Simple"),
        (@formula(y ~ x + group), "Mixed"),
        (@formula(y ~ x * group), "Interaction"),
    ]
    
    for (formula, description) in test_cases
        println("\n🔍 Testing $description: $formula")
        
        try
            model = lm(formula, df)
            
            # Test 1: Pre-compiled approach (should be zero allocation)
            println("  📊 Pre-compiled approach:")
            compiled = compile_formula(model, data)
            output = Vector{Float64}(undef, length(compiled))
            
            # Warmup
            modelrow!(output, compiled, data, 1)
            
            # Test allocation
            allocs = @allocated modelrow!(output, compiled, data, 1)
            expected = modelmatrix(model)[1, :]
            correct = isapprox(output, expected, rtol=1e-12)
            
            println("    Allocations: $allocs bytes")
            println("    Correct: $correct")
            println("    Result: $output")
            
            if allocs == 0 && correct
                println("    ✅ Perfect: Zero allocation + correct result")
            elseif correct
                println("    ⚡ Good: Correct result, minimal allocation ($allocs bytes)")
            else
                println("    ❌ Problem: Incorrect result")
            end
            
            # Test 2: Cached approach
            println("  📊 Cached approach:")
            output2 = Vector{Float64}(undef, length(compiled))
            
            # Warmup
            modelrow!(output2, model, data, 1; cache=true)
            
            # Test allocation
            allocs2 = @allocated modelrow!(output2, model, data, 1; cache=true)
            correct2 = isapprox(output2, expected, rtol=1e-12)
            
            println("    Allocations: $allocs2 bytes")
            println("    Correct: $correct2")
            
            if allocs2 <= 64 && correct2  # Allow small cache allocations
                println("    ✅ Good: Minimal allocation + correct result")
            elseif correct2
                println("    ⚡ OK: Correct result, some allocation ($allocs2 bytes)")
            else
                println("    ❌ Problem: Incorrect result")
            end
            
            # Test 3: ModelRowEvaluator
            println("  📊 ModelRowEvaluator approach:")
            evaluator = ModelRowEvaluator(model, df)
            
            # Warmup
            result3 = evaluator(1)
            
            # Test allocation
            allocs3 = @allocated evaluator(1)
            correct3 = isapprox(result3, expected, rtol=1e-12)
            
            println("    Allocations: $allocs3 bytes")
            println("    Correct: $correct3")
            
            if allocs3 == 0 && correct3
                println("    ✅ Perfect: Zero allocation + correct result")
            elseif correct3
                println("    ⚡ Good: Correct result, minimal allocation ($allocs3 bytes)")
            else
                println("    ❌ Problem: Incorrect result")
            end
            
        catch e
            println("  ❌ Test failed: $e")
        end
    end
    
    println("\n" ^ 1)
    println("🎯 UPDATED MODELROW! SYSTEM TEST COMPLETE")
    
    # Test multi-row functionality
    println("\n🔍 Testing Multi-Row Functionality:")
    try
        model = lm(@formula(y ~ x * group), df)
        compiled = compile_formula(model, data)
        
        # Test matrix version
        row_indices = [1, 2, 3]
        matrix = Matrix{Float64}(undef, length(row_indices), length(compiled))
        
        # Warmup
        modelrow!(matrix, compiled, data, row_indices)
        
        # Test allocation
        alloc = @allocated modelrow!(matrix, compiled, data, row_indices)
        
        # Check correctness
        expected_matrix = modelmatrix(model)
        correct = isapprox(matrix, expected_matrix, rtol=1e-12)
        
        println("  Matrix evaluation:")
        println("    Allocations: $alloc bytes")
        println("    Correct: $correct")
        
        if alloc <= 100 && correct  # Allow small view allocations
            println("    ✅ Multi-row evaluation working well!")
        else
            println("    ⚡ Multi-row evaluation needs optimization")
        end
        
    catch e
        println("  ❌ Multi-row test failed: $e")
    end
end
