# step4_profiling.jl
# Diagnostic tools to identify exactly where allocations are coming from

using Profile, BenchmarkTools

"""
    profile_allocations_detailed(formula, df, data)

Detailed allocation profiling to identify exactly where allocations come from.
"""
function profile_allocations_detailed(formula, df, data)
    println("="^80)
    println("DETAILED ALLOCATION PROFILING FOR: $formula")
    println("="^80)
    
    # Compile both versions
    model = fit(LinearModel, formula, df)
    current_compiled = compile_formula(model, data)
    specialized_compiled = compile_formula_specialized_complete(model, data)
    
    # Pre-allocate outputs
    output_current = Vector{Float64}(undef, length(current_compiled))
    output_specialized = Vector{Float64}(undef, length(specialized_compiled))
    
    # Warmup
    for i in 1:10
        current_compiled(output_current, data, 1)
        specialized_compiled(output_specialized, data, 1)
    end
    
    println("\n1. BASELINE ALLOCATION COMPARISON")
    println("-" ^ 1)
    
    # Current implementation allocations
    current_allocs = @allocated begin
        for i in 1:100
            current_compiled(output_current, data, 1)
        end
    end
    
    # Specialized implementation allocations  
    specialized_allocs = @allocated begin
        for i in 1:100
            specialized_compiled(output_specialized, data, 1)
        end
    end
    
    println("Current: $(current_allocs/100) bytes per call")
    println("Specialized: $(specialized_allocs/100) bytes per call")
    
    if specialized_allocs > 0
        println("\n2. DETAILED ALLOCATION BREAKDOWN")
        println("-")
        
        # Profile specialized execution step by step
        profile_specialized_execution_steps(specialized_compiled, output_specialized, data)
        
        println("\n3. ALLOCATION HOTSPOT ANALYSIS")
        println("-")
        
        # Use allocation profiling to find hotspots
        profile_allocation_hotspots(specialized_compiled, output_specialized, data)
        
    else
        println("🎉 ZERO ALLOCATIONS - No profiling needed!")
    end
    
    return specialized_allocs / 100
end

"""
    profile_specialized_execution_steps(compiled, output, data)

Profile each step of specialized execution to isolate allocation sources.
"""
function profile_specialized_execution_steps(compiled, output, data)
    
    # Test individual components if possible
    formula_data = compiled.data
    formula_op = compiled.operations
    
    println("Testing individual operation types:")
    
    # Test constants
    const_allocs = @allocated begin
        for i in 1:100
            execute_complete_constant_operations!(formula_data.constants, output, data, 1)
        end
    end
    println("  Constants: $(const_allocs/100) bytes per call")
    
    # Test continuous  
    cont_allocs = @allocated begin
        for i in 1:100
            execute_complete_continuous_operations!(formula_data.continuous, output, data, 1)
        end
    end
    println("  Continuous: $(cont_allocs/100) bytes per call")
    
    # Test categorical
    cat_allocs = @allocated begin
        for i in 1:100
            execute_categorical_operations!(formula_data.categorical, output, data, 1)
        end
    end
    println("  Categorical: $(cat_allocs/100) bytes per call")
    
    # Test functions
    if !isempty(formula_data.functions)
        func_allocs = @allocated begin
            for i in 1:100
                execute_linear_function_operations!(formula_data.functions, formula_data.function_scratch, output, data, 1)
            end
        end
        println("  Functions: $(func_allocs/100) bytes per call")
    else
        println("  Functions: 0 bytes per call (no functions)")
    end
    
    # Test interactions
    if !isempty(formula_data.interactions)
        int_allocs = @allocated begin
            for i in 1:100
                execute_interaction_operations!(formula_data.interactions, formula_data.interaction_scratch, output, data, 1)
            end
        end
        println("  Interactions: $(int_allocs/100) bytes per call")
    else
        println("  Interactions: 0 bytes per call (no interactions)")
    end
    
    # Test buffer access
    buffer_allocs = @allocated begin
        for i in 1:100
            fs = formula_data.function_scratch
            is = formula_data.interaction_scratch
        end
    end
    println("  Buffer access: $(buffer_allocs/100) bytes per call")
end

"""
    profile_allocation_hotspots(compiled, output, data)

Use Profile.jl to identify allocation hotspots.
"""
function profile_allocation_hotspots(compiled, output, data)
    
    # Clear any existing profile data
    Profile.clear()
    
    # Enable allocation tracking
    Profile.Allocs.clear()
    
    println("Running allocation profiler...")
    
    # Profile with allocation tracking
    Profile.Allocs.@profile sample_rate=1.0 begin
        for i in 1:1000
            compiled(output, data, ((i-1) % length(data.x)) + 1)
        end
    end
    
    # Get allocation results
    allocs = Profile.Allocs.fetch()
    
    if length(allocs) > 0
        println("Top allocation sources:")
        
        # Group allocations by function
        alloc_by_func = Dict{String, Int}()
        
        for alloc in allocs
            if alloc.size > 0
                func_name = string(alloc.type)
                alloc_by_func[func_name] = get(alloc_by_func, func_name, 0) + alloc.size
            end
        end
        
        # Sort by allocation size
        sorted_allocs = sort(collect(alloc_by_func), by=x->x[2], rev=true)
        
        for (func, size) in sorted_allocs[1:min(10, length(sorted_allocs))]
            println("  $func: $size bytes")
        end
        
        # Also show stack traces for largest allocations
        println("\nLargest individual allocations:")
        large_allocs = filter(a -> a.size >= 32, allocs)
        for (i, alloc) in enumerate(large_allocs[1:min(5, length(large_allocs))])
            println("  Allocation $i: $(alloc.size) bytes of $(alloc.type)")
            if !isempty(alloc.stacktrace)
                println("    at $(alloc.stacktrace[1])")
            end
        end
    else
        println("No allocations detected in profiler")
    end
end

"""
    profile_interaction_components_individually(interaction_data, scratch, output, data, row_idx)

Profile each interaction component individually to isolate issues.
"""
function profile_interaction_components_individually(interaction_data, scratch, output, data, row_idx)
    println("\nPROFILING INTERACTION COMPONENTS INDIVIDUALLY:")
    println("-" ^ 5)
    
    for (i, component) in enumerate(interaction_data.components)
        component_allocs = @allocated begin
            for j in 1:100
                evaluate_unified_component!(component, scratch, data, row_idx)
            end
        end
        
        println("  Component $i ($(component.component_type)): $(component_allocs/100) bytes per call")
        
        # If this component allocates, dive deeper
        if component_allocs > 0
            println("    ⚠️  ALLOCATING COMPONENT DETECTED")
            if component.component_type === :function
                println("    Function data type: $(typeof(component.optimized_data))")
                func_data = component.optimized_data
                if hasfield(typeof(func_data), :scratch_size)
                    println("    Function scratch size: $(func_data.scratch_size)")
                end
            end
        end
    end
end

"""
    run_comprehensive_allocation_profiling()

Run allocation profiling on all the key test cases to identify patterns.
"""
function run_comprehensive_allocation_profiling()
    # Create test data
    n = 200
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,
        w = randn(n),
        group3 = categorical(rand(["A", "B", "C"], n)),           
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      
        binary = categorical(rand(["Yes", "No"], n)),
        response = randn(n)
    )
    data = Tables.columntable(df)
    
    # Profile key test cases
    critical_formulas = [
        (@formula(response ~ x * y), "Simple 2-way continuous"),
        (@formula(response ~ x * group3), "Continuous × Categorical"),  
        (@formula(response ~ log(z) * group4), "Function × Categorical"),
        (@formula(response ~ x * y * group3 + log(z) * group4), "Your target formula"),
        (@formula(response ~ group3 * binary), "Pure categorical"),
        (@formula(response ~ x * y * z), "3-way continuous"),
    ]
    
    allocation_results = Dict{String, Float64}()
    
    for (formula, description) in critical_formulas
        println("\n" * "="^80)
        println("PROFILING: $description")
        println("Formula: $formula")
        println("="^80)
        
        try
            allocs_per_call = profile_allocations_detailed(formula, df, data)
            allocation_results[description] = allocs_per_call
        catch e
            println("❌ Error profiling $description: $e")
            allocation_results[description] = -1.0
        end
    end
    
    # Summary
    println("\n" * "="^80)
    println("ALLOCATION PROFILING SUMMARY")
    println("="^80)
    
    for (desc, allocs) in allocation_results
        if allocs >= 0
            println("$desc: $(allocs) bytes per call")
        else
            println("$desc: Error during profiling")
        end
    end
    
    return allocation_results
end

# Test 4: Field access
struct TestStruct
    field1::Vector{Float64}
    field2::Vector{Float64}
end

"""
    test_buffer_allocation_patterns()

Test specific buffer allocation patterns to identify the source.
"""
function test_buffer_allocation_patterns()
    println("TESTING BUFFER ALLOCATION PATTERNS")
    println("="^50)
    
    # Test 1: Vector creation
    println("1. Testing Vector{Float64} creation:")
    allocs1 = @allocated begin
        for i in 1:100
            v = Vector{Float64}(undef, 10)
        end
    end
    println("   Vector{Float64}(undef, 10) × 100: $allocs1 bytes")
    
    # Test 2: Empty vector creation
    allocs2 = @allocated begin
        for i in 1:100
            v = Float64[]
        end
    end
    println("   Float64[] × 100: $allocs2 bytes")
    
    # Test 3: View creation
    base_vec = Vector{Float64}(undef, 100)
    allocs3 = @allocated begin
        for i in 1:100
            v = view(base_vec, 1:10)
        end
    end
    println("   view(vec, 1:10) × 100: $allocs3 bytes")
    
    test_obj = TestStruct(Vector{Float64}(undef, 10), Vector{Float64}(undef, 10))
    
    allocs4 = @allocated begin
        for i in 1:100
            f1 = test_obj.field1
            f2 = test_obj.field2
        end
    end
    println("   Field access × 100: $allocs4 bytes")
    
    return (allocs1, allocs2, allocs3, allocs4)
end

# Export profiling functions
export profile_allocations_detailed, run_comprehensive_allocation_profiling,
       test_buffer_allocation_patterns, profile_interaction_components_individually

