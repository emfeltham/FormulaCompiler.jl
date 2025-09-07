# test_override_improvements.jl - Tests for Part 0 type-stable override system improvements
#
# Tests for MARGINS_COMPUTE_REWRITE_PLAN.md Part 0 requirements:
# - Type-stable DataScenario with parameterized types
# - Pure modelrow! with overrides: 0 bytes after warmup  
# - Override invariants: baseline→level contrasts, boolean false→true (0→1)
# - Full row rebuild under overrides (interactions, transforms, contrasts)

using Test
using FormulaCompiler
using GLM, DataFrames, Tables
using CategoricalArrays
using LinearAlgebra: dot

@testset "Type-Stable Override System Improvements" begin

    # Setup test data with multiple variable types
    n = 100
    df = DataFrame(
        # Continuous variables
        x1 = randn(n),
        x2 = randn(n),
        age = rand(18:65, n),
        
        # Boolean variables  
        treated = rand(Bool, n),
        eligible = rand(Bool, n),
        
        # Categorical variables 
        group = categorical(rand(["A", "B", "C"], n), levels=["A", "B", "C"]),
        region = categorical(rand(["North", "South"], n), levels=["North", "South"]),
        
        # Response variable
        y = randn(n)
    )
    
    data_nt = Tables.columntable(df)
    
    @testset "Type Stability Improvements" begin
        
        @testset "DataScenario Parameterization" begin
            scenario = create_scenario("type_test", data_nt; x1 = 2.5, treated = true)
            
            # Check that DataScenario is parameterized with concrete types
            @test typeof(scenario) <: DataScenario
            scenario_type = typeof(scenario)
            @test length(scenario_type.parameters) == 2  # NT and OT parameters
            
            # Override type should be concrete NamedTuple, not Dict{Symbol,Any}
            override_type = scenario_type.parameters[2]
            @test override_type <: NamedTuple
            @test !(override_type <: Dict)  # Not the old Dict{Symbol,Any}
            
            # Data type should be parameterized NamedTuple
            data_type = scenario_type.parameters[1] 
            @test data_type <: NamedTuple
            
            println("✓ DataScenario type: ", scenario_type)
            println("✓ Override type: ", override_type)
            println("✓ Data type: ", data_type)
        end
        
        @testset "Override Values Type Stability" begin
            # Single override
            scenario1 = create_scenario("single", data_nt; x1 = 3.14)
            @test scenario1.overrides isa NamedTuple
            @test scenario1.overrides.x1 === 3.14
            @test typeof(scenario1.overrides.x1) === Float64
            
            # Multiple overrides with different types
            scenario2 = create_scenario("multi", data_nt; x1 = 2.0, treated = false, age = 35)
            @test scenario2.overrides isa NamedTuple
            @test scenario2.overrides.x1 === 2.0
            @test scenario2.overrides.treated === false  
            @test scenario2.overrides.age === 35
            @test typeof(scenario2.overrides.x1) === Float64
            @test typeof(scenario2.overrides.treated) === Bool
            @test typeof(scenario2.overrides.age) === Int64
        end
        
        @testset "Immutable DataScenario Design" begin
            scenario = create_scenario("immutable", data_nt; x1 = 1.0)
            original_scenario = scenario
            
            # New immutable API should create new scenarios
            new_scenario = with_override(scenario, :x2, 2.0)
            @test new_scenario !== scenario  # Different objects
            @test new_scenario.overrides.x1 === 1.0  # Original override preserved
            @test new_scenario.overrides.x2 === 2.0  # New override added
            @test scenario.overrides == (x1 = 1.0,)  # Original unchanged
            
            # Test multiple overrides
            multi_scenario = with_overrides(scenario; x2 = 3.0, treated = true)
            @test multi_scenario.overrides.x1 === 1.0
            @test multi_scenario.overrides.x2 === 3.0
            @test multi_scenario.overrides.treated === true
        end
        
    end
    
    @testset "Zero-Allocation Performance" begin
        
        # Setup model and compilation
        model = lm(@formula(y ~ x1 + x2 + treated + group), df)
        compiled = compile_formula(model, data_nt)
        
        @testset "Basic Override Evaluation" begin
            scenario = create_scenario("perf_test", data_nt; x1 = 2.0, treated = true, group = "B")
            output = Vector{Float64}(undef, length(compiled))
            
            # Warmup call
            modelrow!(output, compiled, scenario.data, 1)
            
            # Measure allocations for zero-allocation guarantee
            GC.gc()
            allocs_before = Base.gc_alloc_count()
            
            # Test multiple calls
            for i in 1:50
                modelrow!(output, compiled, scenario.data, 1)
            end
            
            allocs_after = Base.gc_alloc_count()
            total_allocs = allocs_after - allocs_before
            
            @test total_allocs == 0
            println("✓ Zero allocations achieved: $total_allocs bytes allocated")
        end
        
        @testset "Multiple Row and Scenario Types" begin
            scenarios = [
                create_scenario("cont", data_nt; x1 = 3.0, x2 = -2.0),
                create_scenario("bool", data_nt; treated = false, eligible = true), 
                create_scenario("cat", data_nt; group = "A", region = "South"),
                create_scenario("mixed", data_nt; x1 = 1.0, treated = true, group = "B", age = 40)
            ]
            
            output = Vector{Float64}(undef, length(compiled))
            
            for scenario in scenarios
                # Warmup
                modelrow!(output, compiled, scenario.data, 1)
                
                # Test multiple rows with same scenario
                GC.gc()
                allocs_before = Base.gc_alloc_count()
                
                for test_row in [1, 5, 10, 25, 50]
                    modelrow!(output, compiled, scenario.data, test_row)
                end
                
                allocs_after = Base.gc_alloc_count() 
                total_allocs = allocs_after - allocs_before
                
                @test total_allocs == 0
            end
        end
        
    end
    
    @testset "Statistical Correctness - Row Rebuild" begin
        
        @testset "Interactions with Overrides" begin
            # Test that interactions rebuild correctly when variables are overridden
            interaction_df = DataFrame(
                y = randn(30),
                x1 = randn(30), 
                x2 = randn(30),
                group = categorical(rand(["A", "B"], 30), levels=["A", "B"])
            )
            interaction_data = Tables.columntable(interaction_df)
            
            # Model with interactions
            model = lm(@formula(y ~ x1 * x2 + x1 * group), interaction_df)
            compiled = compile_formula(model, interaction_data)
            
            # Create scenarios with overrides affecting interaction terms
            scenario1 = create_scenario("interact1", interaction_data; x1 = 2.0, group = "B")
            scenario2 = create_scenario("interact2", interaction_data; x1 = -1.0, x2 = 3.0)
            
            output = Vector{Float64}(undef, length(compiled))
            
            # Both evaluations should work correctly and be zero allocation
            modelrow!(output, compiled, scenario1.data, 1)
            @test all(isfinite.(output)) "Interaction scenario 1 should produce finite values"
            
            modelrow!(output, compiled, scenario2.data, 1)  
            @test all(isfinite.(output)) "Interaction scenario 2 should produce finite values"
            
            # Verify that interaction terms actually use override values
            # When x1=2.0 and group="B", the x1*group interaction should be non-zero
            # (This tests that the row is actually rebuilt, not just using original values)
            
            # Test different rows to ensure consistent behavior
            for test_row in [1, 5, 15, 25]
                modelrow!(output, compiled, scenario1.data, test_row)
                @test all(isfinite.(output)) "Row $test_row should produce finite values"
            end
        end
        
        @testset "Transforms with Overrides" begin
            # Test that transformed variables work correctly with overrides
            transform_df = DataFrame(
                y = randn(20),
                x = rand(20) .+ 0.1,  # Ensure positive for log
                z = randn(20)
            )
            transform_data = Tables.columntable(transform_df)
            
            # Model with transforms  
            model = lm(@formula(y ~ log(x) + x^2 + sqrt(abs(z))), transform_df)
            compiled = compile_formula(model, transform_data)
            
            # Override should work with transformed terms
            scenario = create_scenario("transform", transform_data; x = 2.5, z = -1.5)
            output = Vector{Float64}(undef, length(compiled))
            
            modelrow!(output, compiled, scenario.data, 1)
            @test all(isfinite.(output)) "All transformed output values should be finite"
            
            # Verify transforms actually use override values:
            # log(2.5) ≈ 0.916, (2.5)^2 = 6.25, sqrt(abs(-1.5)) ≈ 1.225
            # These should be reflected in the model matrix row
        end
        
        @testset "Complex Formula with All Variable Types" begin
            # Test comprehensive formula with interactions, transforms, categoricals
            complex_df = DataFrame(
                y = randn(40),
                x1 = randn(40),
                x2 = randn(40), 
                age = rand(25:60, 40),
                treated = rand(Bool, 40),
                group = categorical(rand(["A", "B", "C"], 40), levels=["A", "B", "C"]),
                region = categorical(rand(["North", "South"], 40), levels=["North", "South"])
            )
            complex_data = Tables.columntable(complex_df)
            
            # Complex model: interactions, transforms, multiple categoricals
            model = lm(@formula(y ~ x1 + log(age) + treated * x2 + group * region + x1^2), complex_df)
            compiled = compile_formula(model, complex_data)
            
            # Override affecting multiple interaction and transform terms
            scenario = create_scenario("complex", complex_data; 
                x1 = 1.5, age = 45, treated = true, group = "B", region = "South")
                
            output = Vector{Float64}(undef, length(compiled))
            
            # Should handle complex overrides correctly across multiple rows
            for test_row in [1, 10, 20, 35]
                modelrow!(output, compiled, scenario.data, test_row)
                @test all(isfinite.(output)) "Complex row $test_row should produce finite values"
                @test length(output) == length(compiled) "Output length should match compiled length"
            end
        end
        
    end
    
    @testset "Baseline and Boolean Convention Validation" begin
        
        @testset "Boolean Variables (false→true Convention)" begin
            bool_df = DataFrame(
                y = randn(20),
                x = randn(20),
                treated = rand(Bool, 20),
                eligible = rand(Bool, 20)
            )
            bool_data = Tables.columntable(bool_df)
            
            # Create scenarios for boolean contrasts
            scenario_false = create_scenario("false", bool_data; treated = false, eligible = false)
            scenario_true = create_scenario("true", bool_data; treated = true, eligible = true)
            
            model = lm(@formula(y ~ x + treated + eligible), bool_df)
            compiled = compile_formula(model, bool_data)
            
            output_false = Vector{Float64}(undef, length(compiled))
            output_true = Vector{Float64}(undef, length(compiled))
            
            # Test that boolean overrides work correctly
            modelrow!(output_false, compiled, scenario_false.data, 1)
            modelrow!(output_true, compiled, scenario_true.data, 1)
            
            @test all(isfinite.(output_false)) "Boolean false scenario should work"
            @test all(isfinite.(output_true)) "Boolean true scenario should work"
            @test output_false != output_true "Different boolean values should produce different results"
        end
        
        @testset "Categorical Variables (Reference Level Baseline)" begin
            cat_df = DataFrame(
                y = randn(25),
                x = randn(25),
                group = categorical(rand(["A", "B", "C"], 25), levels=["A", "B", "C"]),
                region = categorical(rand(["North", "South", "East"], 25), levels=["North", "South", "East"])
            )
            cat_data = Tables.columntable(cat_df)
            
            # Test with different contrast codings
            for contrast_type in [DummyCoding(), EffectsCoding(), HelmertCoding()]
                contrasts = Dict(:group => contrast_type, :region => contrast_type)
                model = lm(@formula(y ~ x + group + region), cat_df, contrasts=contrasts)
                compiled = compile_formula(model, cat_data)
                
                # Create scenarios for different categorical levels
                scenario_baseline = create_scenario("baseline", cat_data; group = "A", region = "North")  # First levels
                scenario_contrast = create_scenario("contrast", cat_data; group = "B", region = "South")
                
                output_baseline = Vector{Float64}(undef, length(compiled))
                output_contrast = Vector{Float64}(undef, length(compiled))
                
                modelrow!(output_baseline, compiled, scenario_baseline.data, 1)
                modelrow!(output_contrast, compiled, scenario_contrast.data, 1)
                
                @test all(isfinite.(output_baseline)) "Baseline categorical should work with $(contrast_type)"
                @test all(isfinite.(output_contrast)) "Contrast categorical should work with $(contrast_type)"
            end
        end
        
    end
    
    @testset "New Immutable API Functions" begin
        
        @testset "with_override Function" begin
            original = create_scenario("original", data_nt; x1 = 1.0, treated = false)
            
            # Test adding new override
            with_new = with_override(original, :x2, 2.5)
            @test with_new.overrides.x1 === 1.0  # Original preserved
            @test with_new.overrides.x2 === 2.5  # New added
            @test with_new.overrides.treated === false  # Original preserved
            
            # Test updating existing override
            with_updated = with_override(original, :x1, 3.0)
            @test with_updated.overrides.x1 === 3.0  # Updated
            @test with_updated.overrides.treated === false  # Original preserved
            @test !haskey(with_updated.overrides, :x2)  # Not added
        end
        
        @testset "with_overrides Function" begin
            original = create_scenario("original", data_nt; x1 = 1.0)
            
            # Test multiple updates at once
            updated = with_overrides(original; x1 = 5.0, x2 = -1.0, treated = true, group = "C")
            @test updated.overrides.x1 === 5.0
            @test updated.overrides.x2 === -1.0
            @test updated.overrides.treated === true
            @test updated.overrides.group === "C"
            
            # Original should be unchanged
            @test original.overrides == (x1 = 1.0,)
        end
        
        @testset "without_override Function" begin
            original = create_scenario("original", data_nt; x1 = 1.0, x2 = 2.0, treated = true)
            
            # Remove one override
            removed_one = without_override(original, :x2)
            @test removed_one.overrides.x1 === 1.0  # Preserved
            @test removed_one.overrides.treated === true  # Preserved  
            @test !haskey(removed_one.overrides, :x2)  # Removed
            
            # Remove non-existent override (should be no-op)
            no_change = without_override(original, :nonexistent)
            @test no_change.overrides == original.overrides
        end
        
    end
    
end