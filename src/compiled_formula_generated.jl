# src/compiled_formula_generated.jl
# Updated @generated interface using compositional backend

###############################################################################
# Cache and Interface (Keep the Good Parts)
###############################################################################

# Keep the caching mechanism but adapt for compositional functions
const FORMULA_CACHE = Dict{UInt64, Tuple{Symbol, Vector{Symbol}, Int}}()

"""
    compile_formula_generated(model) -> (formula_val, output_width, column_names)

Main interface that uses compositional backend but maintains @generated dispatch.
"""
function compile_formula_generated(model)
    # Register the formula and return the Val type for @generated dispatch
    formula_val = register_formula_compositional!(model)
    
    # Get metadata for the compiled formula
    func_name, column_names, output_width = FORMULA_CACHE[typeof(formula_val).parameters[1]]
    
    return (formula_val, output_width, column_names)
end

"""
    register_formula_compositional!(model) -> Val{formula_hash}

Register formula using compositional compiler and return Val type for @generated dispatch
"""
function register_formula_compositional!(model)
    formula_hash = hash(string(fixed_effects_form(model).rhs))
    
    # Use compositional compiler as backend
    func_name, output_width, column_names = compile_formula_compositional_efficient(model)
    
    # Store in cache for @generated dispatch
    FORMULA_CACHE[formula_hash] = (func_name, column_names, output_width)
    
    return Val(formula_hash)
end

"""
    modelrow!(row_vec, ::Val{formula_hash}, data, row_idx)

@generated function for zero-allocation formula evaluation using compositional backend
"""
@generated function modelrow!(row_vec, ::Val{formula_hash}, data, row_idx) where formula_hash
    # Get the compositional function from cache
    func_name, column_names, output_width = FORMULA_CACHE[formula_hash]
    
    # Generate code that calls the precompiled compositional function
    return quote
        $func_name(row_vec, data, row_idx)
    end
end

###############################################################################
# Convenience Functions
###############################################################################

"""
    get_compiled_function(model) -> Function

Get the compiled function directly without @generated overhead
"""
function get_compiled_function(model)
    func_name, output_width, column_names = compile_formula_compositional_efficient(model)
    return getproperty(Main, func_name), output_width, column_names
end

"""
    compile_and_benchmark(model, data, n_trials=1000)

Compile formula and benchmark performance
"""
function compile_and_benchmark(model, data, n_trials=1000)
    # Get compiled function
    func, output_width, column_names = get_compiled_function(model)
    
    # Setup
    row_vec = Vector{Float64}(undef, output_width)
    
    println("=== Benchmarking Compiled Formula ===")
    println("Output width: $output_width")
    println("Columns: $column_names")
    
    # Warmup
    func(row_vec, data, 1)
    
    # Benchmark single evaluation
    println("\nSingle evaluation:")
    @time func(row_vec, data, 1)
    
    # Check allocations
    allocs = @allocated func(row_vec, data, 1)
    println("Allocations: $allocs bytes")
    
    # Benchmark multiple evaluations
    println("\n$n_trials evaluations:")
    @time for i in 1:n_trials
        func(row_vec, data, (i % length(data[1])) + 1)
    end
    
    total_allocs = @allocated for i in 1:n_trials
        func(row_vec, data, (i % length(data[1])) + 1)
    end
    println("Total allocations: $total_allocs bytes")
    println("Per evaluation: $(total_allocs / n_trials) bytes")
    
    return func, output_width, column_names
end

###############################################################################
# Example Usage and Testing
###############################################################################

"""
Example usage:

```julia
using GLM, DataFrames, CategoricalArrays
df = DataFrame(
    x = randn(1000), 
    y = randn(1000), 
    z = abs.(randn(1000)) .+ 0.1,
    group = categorical(rand(["A", "B"], 1000))
)

# Complex formula that would fail with old system
model = lm(@formula(y ~ x + x^2 * log(z) + group), df)

# Method 1: @generated interface (backward compatible)
formula_val, output_width, column_names = compile_formula_generated(model)
row_vec = Vector{Float64}(undef, output_width)
data = Tables.columntable(df)

@btime modelrow!(row_vec, formula_val, data, 1)  # Should be ~50-100ns, 0 alloc

# Method 2: Direct function call (maximum performance)
func, output_width, column_names = get_compiled_function(model)
@btime func(row_vec, data, 1)  # Should be ~30-80ns, 0 alloc

# Method 3: Compile and benchmark
compile_and_benchmark(model, data)
```
"""

function test_generated_interface()
    println("=== Testing Updated @generated Interface ===")

    
    # Create test data
    n = 100
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B"], n))
    )
    
    # Test the complex formula that used to fail
    model = lm(@formula(y ~ x + x^2 * log(z) + group), df)
    data = Tables.columntable(df)
    
    println("Testing @generated interface...")
    
    # Test @generated interface
    formula_val, output_width, column_names = compile_formula_generated(model)
    row_vec = Vector{Float64}(undef, output_width)
    
    println("✅ Compilation successful")
    println("Output width: $output_width")
    println("Columns: $column_names")
    
    # Test evaluation
    modelrow!(row_vec, formula_val, data, 1)
    println("✅ @generated evaluation successful")
    
    # Test direct function
    func, _, _ = get_compiled_function(model)
    row_vec2 = Vector{Float64}(undef, output_width)
    func(row_vec2, data, 1)
    println("✅ Direct function evaluation successful")
    
    # Verify both give same results
    if isapprox(row_vec, row_vec2, atol=1e-15)
        println("✅ Both interfaces give identical results")
    else
        println("❌ Interfaces give different results!")
    end
    
    # Verify against model matrix
    mm_row = modelmatrix(model)[1, :]
    if isapprox(mm_row, row_vec, atol=1e-10)
        println("✅ Results match model matrix!")
    else
        println("❌ Results don't match model matrix:")
        println("   Expected: $mm_row")
        println("   Got:      $row_vec")
    end
    
    # Quick performance check
    println("\nPerformance check:")
    println("@generated interface:")
    @time modelrow!(row_vec, formula_val, data, 1)
    
    println("Direct function:")
    @time func(row_vec2, data, 1)
    
    println("Allocations check:")
    allocs1 = @allocated modelrow!(row_vec, formula_val, data, 1)
    allocs2 = @allocated func(row_vec2, data, 1)
    println("@generated: $allocs1 bytes")
    println("Direct:     $allocs2 bytes")
    
    return formula_val, func, output_width, column_names
end
