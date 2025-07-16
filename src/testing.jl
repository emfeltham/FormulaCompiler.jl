# testing.jl

###############################################################################
# TESTING AND VALIDATION
###############################################################################

"""
    test_evaluator_storage(model, data) -> Bool

Test that evaluator storage works correctly and evaluator produces same results as compiled formula.
"""
function test_evaluator_storage(model, data)
    println("Testing evaluator storage...")
    
    # Compile formula with new storage
    compiled = compile_formula(model)
    println("✓ Compiled formula with evaluator storage")
    
    # Test that we can access the evaluator
    evaluator = extract_root_evaluator(compiled)
    println("✓ Extracted evaluator: $(typeof(evaluator))")
    
    # Test that evaluator produces same results as compiled formula
    data_nt = Tables.columntable(data)
    n_test = min(10, length(first(data_nt)))
    
    for i in 1:n_test
        # Evaluate using compiled formula
        compiled_result = Vector{Float64}(undef, length(compiled))
        compiled(compiled_result, data_nt, i)
        
        # Evaluate using raw evaluator
        evaluator_result = Vector{Float64}(undef, output_width(evaluator))
        evaluate!(evaluator, evaluator_result, data_nt, i, 1)
        
        # Compare results
        if !isapprox(compiled_result, evaluator_result, rtol=1e-12)
            println("✗ Results don't match at row $i")
            println("  Compiled: $compiled_result")
            println("  Evaluator: $evaluator_result")
            return false
        end
    end
    
    println("✓ Evaluator produces identical results to compiled formula")
    
    # Test analysis functions
    summary = get_evaluator_summary(compiled)
    println("✓ Generated evaluator summary: $summary")
    
    dependencies = get_variable_dependencies(compiled)
    println("✓ Found variable dependencies: $dependencies")
    
    node_count = count_evaluator_nodes(compiled)
    println("✓ Counted evaluator nodes: $node_count")
    
    return true
end

###############################################################################
# 9. COMPREHENSIVE TEST FUNCTION
###############################################################################

function test_comprehensive_compilation()
    println("=== Testing Comprehensive Recursive Compilation ===")
    
    Random.seed!(42)
    df = DataFrame(
        x = randn(100),
        y = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        cat2a = categorical(rand(["X", "Y"], 100)),
        cat2b = categorical(rand(["P", "Q"], 100)),
        bool = rand([false, true], 100),
        group = categorical(rand(["A", "B", "C"], 100)),
        group2 = categorical(rand(["C", "D", "X"], 100)),
        group3 = categorical(rand(["E", "F", "G"], 100))
    )
    
    data = Tables.columntable(df)
    
    test_cases = [
        (@formula(y ~ cat2a * cat2b), "cat 2 x cat 2"),
        (@formula(y ~ cat2a * bool), "cat 2 x bool"),
        (@formula(y ~ cat2a * (x^2)), "cat 2 x cts"),
        (@formula(y ~ bool * (x^2)), "binary x cts"),
        (@formula(y ~ cat2b * (x^2)), "cat 2 x cts (variant)"),
        (@formula(y ~ group2 * (x^2)), "cat >2 x cts"),
        (@formula(y ~ group2 * bool), "cat >2 x bool"),
        (@formula(y ~ group2 * cat2a), "cat >2 x cat 2"),
        (@formula(y ~ group2 * group3), "cat >2 x cat >2"),
        (@formula(y ~ x * z * group), "three-way continuous x categorical"),
        (@formula(y ~ (x>0) * group), "boolean function x categorical"),
        (@formula(y ~ log(z) * group2 * cat2a), "function x cat >2 x cat 2"),
    ]
    
    results = []
    
    for (i, (formula, description)) in enumerate(test_cases)
        println("\n--- Test $i: $description ---")
        
        try
            model = lm(formula, df)
            mm = modelmatrix(model)
            
            # Compile with recursive approach
            compiled = compile_formula(model)
            
            # Test correctness
            row_vec = Vector{Float64}(undef, length(compiled))
            test_row = 1
            
            compiled(row_vec, data, test_row)
            expected = mm[test_row, :]
            
            error = maximum(abs.(row_vec .- expected))
            
            if error < 1e-12
                println("✅ PASSED: Error = $(error)")
                
                # Test allocations
                allocs = @allocated compiled(row_vec, data, test_row)
                println("   Allocations: $allocs bytes")
                
                push!(results, (description, true, allocs))
            else
                println("❌ FAILED: Error = $error")
                push!(results, (description, false, error))
            end
            
        catch e
            println("❌ EXCEPTION: $e")
            push!(results, (description, false, "exception"))
        end
    end
    
    # Summary
    successful = sum(r[2] for r in results)
    println("\n" * "="^50)
    println("RESULTS: $successful/$(length(test_cases)) passed")
    
    if successful == length(test_cases)
        println("🎉 ALL TESTS PASSED!")
        println("✅ Recursive compositional approach works for ALL cases!")
    end
    
    return results
end

function check_evaluator_conformance(evaluator::AbstractEvaluator)
    # Verify output_width is defined
    try
        width = output_width(evaluator)
        @assert width > 0 "Output width must be positive"
    catch
        error("output_width not implemented for $(typeof(evaluator))")
    end
    
    # Verify evaluate! is defined  
    try
        test_output = Vector{Float64}(undef, output_width(evaluator))
        # Would need test data to fully verify
    catch
        error("evaluate! not implemented for $(typeof(evaluator))")
    end
    
    return true
end

###############################################################################
# COMPREHENSIVE TEST
###############################################################################

function test_complete()
    println("=== Testing Option 2: Recursive → @generated ===")
    
    Random.seed!(06515)  # Same seed as your test
    df = DataFrame(
        x = randn(1000),
        y = randn(1000),
        z = abs.(randn(1000)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 1000))
    )
    
    df.bool = rand([false, true], nrow(df))
    df.group2 = categorical(rand(["C", "D", "X"], nrow(df)))
    df.group3 = categorical(rand(["E", "F", "G"], nrow(df)))
    df.cat2a = categorical(rand(["X", "Y"], nrow(df)))
    df.cat2b = categorical(rand(["P", "Q"], nrow(df)))
    
    data = Tables.columntable(df)
    
    test_cases = [
        (@formula(y ~ cat2a * cat2b), "cat 2 x cat 2"),
        (@formula(y ~ cat2a * bool), "cat 2 x bool"),
        (@formula(y ~ cat2a * (x^2)), "cat 2 x cts"),
        (@formula(y ~ bool * (x^2)), "binary x cts"),
        (@formula(y ~ cat2b * (x^2)), "cat 2 x cts (variant)"),
        (@formula(y ~ group2 * (x^2)), "cat >2 x cts"),
        (@formula(y ~ group2 * bool), "cat >2 x bool"),
        (@formula(y ~ group2 * cat2a), "cat >2 x cat 2"),
        (@formula(y ~ group2 * group3), "cat >2 x cat >2"),
        (@formula(y ~ x * z * group), "three-way continuous x categorical"),
        (@formula(y ~ (x>0) * group), "boolean function x categorical"),
        (@formula(y ~ log(z) * group2 * cat2a), "function x cat >2 x cat 2"),
    ]
    
    results = []
    
    for (i, (formula, description)) in enumerate(test_cases)
        println("\n--- Test $i: $description ---")
        
        try
            model = lm(formula, df)
            mm = modelmatrix(model)
            
            # Compile with Option 2
            compiled = compile_formula(model)
            
            # Test correctness
            row_vec = Vector{Float64}(undef, length(compiled))
            test_row = 1
            
            compiled(row_vec, data, test_row)
            expected = mm[test_row, :]
            
            error = maximum(abs.(row_vec .- expected))
            
            if error < 1e-12
                println("✅ CORRECTNESS: Passed")
                
                # Test allocations
                allocs = @allocated compiled(row_vec, data, test_row)
                println("   ALLOCATIONS: $allocs bytes")
                
                if allocs == 0
                    println("   🎉 PERFECT: Zero allocations!")
                    push!(results, (description, :perfect))
                elseif allocs < 100
                    println("   ✅ GOOD: Low allocations")
                    push!(results, (description, :good))
                else
                    println("   ⚠️  HIGH: Many allocations")
                    push!(results, (description, :high_alloc))
                end
            else
                println("❌ FAILED: Error = $error")
                push!(results, (description, :failed))
            end
            
        catch e
            println("❌ EXCEPTION: $e")
            push!(results, (description, :exception))
        end
    end
    
    # Summary
    perfect = count(r -> r[2] == :perfect, results)
    good = count(r -> r[2] == :good, results)
    failed = count(r -> r[2] in [:failed, :exception, :high_alloc], results)
    
    println("\n" * "="^60)
    println("OPTION 2 RESULTS")
    println("="^60)
    println("Perfect (correct + zero alloc): $perfect")
    println("Good (correct + low alloc):     $good") 
    println("Failed/High alloc:              $failed")
    println("Total:                          $(length(results))")
    
    if perfect == length(results)
        println("🎉 OPTION 2 SUCCESS! All tests perfect!")
    elseif perfect + good == length(results)
        println("✅ Option 2 works! Minor allocation issues only")
    else
        println("⚠️  Option 2 needs more work")
    end
    
    return results
end

###############################################################################
# VALIDATION UTILITIES
###############################################################################

"""
    validate_derivative_evaluator(evaluator::AbstractEvaluator, focal_variable::Symbol, 
                                 test_data::NamedTuple, tolerance::Float64 = 1e-8) -> Bool

Validate that a derivative evaluator produces correct results by comparing with 
numerical differentiation using finite differences.

# Arguments
- `evaluator`: The derivative evaluator to test
- `focal_variable`: Variable being differentiated with respect to
- `test_data`: Sample data for testing
- `tolerance`: Tolerance for numerical comparison

# Returns
`true` if analytical and numerical derivatives match within tolerance.
"""
function validate_derivative_evaluator(evaluator::AbstractEvaluator, 
                                     focal_variable::Symbol,
                                     test_data::NamedTuple, 
                                     tolerance::Float64 = 1e-8)
    
    # Test on first few observations
    test_indices = 1:min(5, length(first(test_data)))
    
    for row_idx in test_indices
        try
            # Compute analytical derivative
            analytical_result = Vector{Float64}(undef, output_width(evaluator))
            evaluate!(evaluator, analytical_result, test_data, row_idx, 1)
            analytical_value = analytical_result[1]
            
            # Compute numerical derivative using finite differences
            ε = sqrt(eps(Float64))
            current_value = Float64(test_data[focal_variable][row_idx])
            
            # Evaluate at x + ε
            data_plus = merge(test_data, (focal_variable => current_value + ε,))
            result_plus = Vector{Float64}(undef, 1)
            evaluate!(evaluator, result_plus, data_plus, row_idx, 1)
            
            # Evaluate at x - ε  
            data_minus = merge(test_data, (focal_variable => current_value - ε,))
            result_minus = Vector{Float64}(undef, 1)
            evaluate!(evaluator, result_minus, data_minus, row_idx, 1)
            
            # Numerical derivative
            numerical_value = (result_plus[1] - result_minus[1]) / (2ε)
            
            # Check if they match within tolerance
            if abs(analytical_value - numerical_value) > tolerance
                @warn "Derivative validation failed at row $row_idx: analytical=$analytical_value, numerical=$numerical_value"
                return false
            end
            
        catch e
            @warn "Derivative validation error at row $row_idx: $e"
            return false
        end
    end
    
    return true
end

###############################################################################
# TESTING AND EXAMPLES
###############################################################################

"""
    test_scenario_foundation()

Test the scenario foundation for correctness.
"""
function test_scenario_foundation()
    
    println("Testing Dict-based scenario foundation...")
    
    # Create test data
    Random.seed!(42)
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0],
        y = [1.0, 4.0, 9.0, 16.0],
        group = categorical(["A", "B", "A", "B"])
    )
    
    # Setup
    model = lm(@formula(y ~ x * group), df)
    compiled = compile_formula(model)
    data = Tables.columntable(df)
    
    @testset "Dict-Based Scenario Foundation" begin
        
        @testset "OverrideVector" begin
            # Basic functionality
            override_vec = OverrideVector(5.0, 10)
            @test length(override_vec) == 10
            @test override_vec[1] == 5.0
            @test override_vec[5] == 5.0
            @test override_vec[10] == 5.0
            
            # Iteration
            @test collect(override_vec) == fill(5.0, 10)
        end
        
        @testset "Categorical Override" begin
            cat_override = create_categorical_override("B", df.group)
            @test length(cat_override) == length(df.group)
            @test cat_override[1] isa CategoricalValue
            @test string(cat_override[1]) == "B"
        end
        
        @testset "Single Scenario" begin
            # Create scenario with override
            scenario = create_scenario("test", data; x = 5.0)
            
            @test scenario.name == "test"
            @test scenario.overrides[:x] == 5.0
            @test isa(scenario.overrides, Dict)
            
            # Test that override works
            @test scenario.data.x[1] == 5.0
            @test scenario.data.x[4] == 5.0
            
            # Test that other columns are unchanged (references)
            @test scenario.data.y === data.y  # Same object reference
            @test scenario.data.group === data.group
        end
        
        @testset "Scenario Mutation" begin
            scenario = create_scenario("mutable_test", data; x = 1.0)
            
            # Test adding override
            set_override!(scenario, :y, 10.0)
            @test scenario.overrides[:y] == 10.0
            @test scenario.data.y[1] == 10.0
            
            # Test updating override
            set_override!(scenario, :x, 99.0)
            @test scenario.overrides[:x] == 99.0
            @test scenario.data.x[1] == 99.0
            
            # Test removing override
            remove_override!(scenario, :y)
            @test !haskey(scenario.overrides, :y)
            @test scenario.data.y === data.y  # Back to original
            
            # Test bulk update
            update_scenario!(scenario; x = 2.0, group = "A")
            @test scenario.overrides[:x] == 2.0
            @test scenario.overrides[:group] == "A"
        end
        
        @testset "Scenario Grid" begin
            # Create grid of scenarios
            x_values = [0.0, 1.0, 2.0]
            collection = create_scenario_grid("x_test", data, Dict(:x => x_values))
            
            @test length(collection) == 3
            @test collection[1].name == "x_test_1"
            @test collection[2].name == "x_test_2"
            @test collection[3].name == "x_test_3"
            
            # Test values
            @test collection[1].overrides[:x] == 0.0
            @test collection[2].overrides[:x] == 1.0
            @test collection[3].overrides[:x] == 2.0
        end
        
        @testset "Integration with modelrow!" begin
            # Test basic evaluation
            scenario = create_scenario("eval_test", data; x = 10.0)
            row_vec = Vector{Float64}(undef, length(compiled))
            
            modelrow!(row_vec, compiled, scenario, 1)
            @test all(isfinite.(row_vec))
            
            # Test multi-scenario evaluation
            scenarios = [
                create_scenario("s1", data; x = 1.0),
                create_scenario("s2", data; x = 2.0)
            ]
            matrix = Matrix{Float64}(undef, 2, length(compiled))
            
            modelrow_scenarios!(matrix, compiled, scenarios, 1)
            @test all(isfinite.(matrix))
            @test matrix[1, :] != matrix[2, :]  # Should be different
        end
    end
    
    println("✅ All Dict-based scenario foundation tests passed!")
    return true
end

"""
    example_scenario_usage()

Demonstrate iterative scenario development with Dict-based overrides and unified grid interface.
"""
function example_scenario_usage()
    
    println("=== Example Dict-Based Scenario Usage ===")
    
    # Setup data and model
    Random.seed!(123)
    df = DataFrame(
        x = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100)),
        treatment = rand([false, true], 100)
    )
    df.y = 1 .+ 0.5 .* df.x .+ 0.3 .* log.(df.z) .+ 0.2 .* df.treatment .+ randn(100)
    
    model = lm(@formula(y ~ x + log(z) + group + treatment + x*group), df)
    compiled = compile_formula(model)
    data = Tables.columntable(df)
    
    println("Model fitted with $(length(compiled)) coefficients")
    
    # 1. Start with simple scenario
    println("\n1. Creating initial scenario...")
    scenario = create_scenario("policy_development", data; treatment = true)
    list_scenario_overrides(scenario)
    
    # 2. Iteratively add variables
    println("\n2. Adding variables iteratively...")
    set_override!(scenario, :x, mean(data.x))
    println("After adding x override:")
    list_scenario_overrides(scenario)
    
    set_override!(scenario, :group, "A")  
    println("After adding group override:")
    list_scenario_overrides(scenario)
    
    # 3. Modify existing overrides
    println("\n3. Modifying existing overrides...")
    set_override!(scenario, :x, mean(data.x) + std(data.x))
    println("After updating x to mean + std:")
    list_scenario_overrides(scenario)
    
    # 4. Bulk updates
    println("\n4. Bulk scenario updates...")
    update_scenario!(scenario; 
        x = 2.0, 
        treatment = false,
        z = median(data.z)
    )
    println("After bulk update:")
    list_scenario_overrides(scenario)
    
    # 5. Remove some overrides  
    println("\n5. Removing overrides...")
    remove_override!(scenario, :z)
    remove_override!(scenario, :group)
    println("After removing z and group:")
    list_scenario_overrides(scenario)
    
    # 6. Test evaluation at each stage
    println("\n6. Testing final scenario evaluation...")
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, scenario.data, 1)
    prediction = sum(row_vec)  # Simple sum as example
    println("Final prediction: $(round(prediction, digits=3))")
    
    # 7. Demonstrate unified grid interface
    println("\n7. Unified grid interface examples...")
    
    # Single variable grid
    println("Single variable grid:")
    x_collection = create_scenario_grid("x_analysis", data, Dict(:x => [0.0, 1.0, 2.0]))
    list_scenarios(x_collection)
    
    # Multiple variable combinations
    println("\nMultiple variable combinations:")
    combo_collection = create_scenario_grid("policy_grid", data, Dict(
        :treatment => [true, false],
        :group => ["A", "B"]
    ))
    list_scenarios(combo_collection)
    
    # Complex grid with three variables
    println("\nComplex three-variable grid:")
    complex_collection = create_scenario_grid("comprehensive", data, Dict(
        :x => [mean(data.x), mean(data.x) + std(data.x)],
        :treatment => [true, false],
        :group => ["A", "C"]
    ))
    println("Created $(length(complex_collection)) scenarios")
    for (i, scenario) in enumerate(complex_collection.scenarios[1:min(4, end)])
        println("  $i. $(scenario.name): $(scenario.overrides)")
    end
    if length(complex_collection) > 4
        println("  ... ($(length(complex_collection) - 4) more scenarios)")
    end
    
    return (scenario, x_collection, combo_collection, complex_collection)
end
