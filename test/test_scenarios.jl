# test/test_scenarios.jl
# Tests for data scenario functionality

@testset "Data Scenarios" begin
    
    # Create test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0, 5.0],
        y = [10.0, 20.0, 30.0, 40.0, 50.0],
        group = categorical(["A", "B", "A", "B", "A"]),
        flag = [true, false, true, false, true]
    )
    data = Tables.columntable(df)
    model = lm(@formula(y ~ x * group + flag), df)
    compiled = compile_formula(model)
    
    @testset "OverrideVector" begin
        # Test basic functionality
        override_vec = OverrideVector(5.0, 10)
        @test length(override_vec) == 10
        @test override_vec[1] == 5.0
        @test override_vec[5] == 5.0
        @test override_vec[10] == 5.0
        
        # Test iteration
        @test collect(override_vec) == fill(5.0, 10)
        
        # Test with different types
        bool_override = OverrideVector(true, 5)
        @test bool_override[1] == true
        @test all(collect(bool_override))
        
        # Test size and indexing
        @test size(override_vec) == (10,)
        @test IndexStyle(typeof(override_vec)) == IndexLinear()
    end
    
    @testset "Categorical Overrides" begin
        # Test categorical override creation
        cat_override = create_categorical_override("B", df.group)
        @test length(cat_override) == length(df.group)
        @test cat_override[1] isa CategoricalValue
        @test string(cat_override[1]) == "B"
        @test string(cat_override[3]) == "B"
        
        # Test error for invalid level
        @test_throws ErrorException create_categorical_override("Z", df.group)
        
        # Test with CategoricalValue input
        cat_val = df.group[2]  # "B"
        cat_override2 = create_categorical_override(cat_val, df.group)
        @test string(cat_override2[1]) == "B"
    end
    
    @testset "Basic Scenario Creation" begin
        # Test scenario with single override
        scenario = create_scenario("test_x", data; x = 10.0)
        @test scenario.name == "test_x"
        @test scenario.overrides[:x] == 10.0
        @test scenario.data.x[1] == 10.0
        @test scenario.data.x[5] == 10.0
        
        # Test that other columns unchanged
        @test scenario.data.y === data.y  # Same reference
        @test scenario.data.group === data.group
        @test scenario.data.flag === data.flag
        
        # Test scenario with multiple overrides
        scenario2 = create_scenario("test_multi", data; x = 99.0, flag = false)
        @test scenario2.overrides[:x] == 99.0
        @test scenario2.overrides[:flag] == false
        @test scenario2.data.x[1] == 99.0
        @test scenario2.data.flag[1] == false
        
        # Test scenario with no overrides
        scenario3 = create_scenario("no_overrides", data)
        @test isempty(scenario3.overrides)
        @test scenario3.data === data  # Same reference
    end
    
    @testset "Scenario Mutation" begin
        scenario = create_scenario("mutable", data; x = 1.0)
        
        # Test adding new override
        set_override!(scenario, :y, 100.0)
        @test haskey(scenario.overrides, :y)
        @test scenario.overrides[:y] == 100.0
        @test scenario.data.y[1] == 100.0
        @test scenario.data.y[3] == 100.0
        
        # Test updating existing override
        set_override!(scenario, :x, 999.0)
        @test scenario.overrides[:x] == 999.0
        @test scenario.data.x[1] == 999.0
        
        # Test removing override
        remove_override!(scenario, :y)
        @test !haskey(scenario.overrides, :y)
        @test scenario.data.y === data.y  # Back to original
        
        # Test bulk update
        update_scenario!(scenario; x = 2.0, flag = false, group = "A")
        @test scenario.overrides[:x] == 2.0
        @test scenario.overrides[:flag] == false
        @test scenario.overrides[:group] == "A"
        @test scenario.data.x[1] == 2.0
        @test scenario.data.flag[1] == false
        @test string(scenario.data.group[1]) == "A"
    end
    
    @testset "Scenario Grid Creation" begin
        # Test single variable grid
        x_values = [0.0, 1.0, 2.0]
        collection = create_scenario_grid("x_test", data, Dict(:x => x_values))
        
        @test length(collection) == 3
        @test collection.name == "x_test"
        @test collection.original_data === data
        
        # Check scenario names and values
        @test collection[1].name == "x_test_x_0.0"
        @test collection[2].name == "x_test_x_1.0"
        @test collection[3].name == "x_test_x_2.0"
        
        @test collection[1].overrides[:x] == 0.0
        @test collection[2].overrides[:x] == 1.0
        @test collection[3].overrides[:x] == 2.0
        
        # Test two variable grid
        combo_collection = create_scenario_grid("combo", data, Dict(
            :x => [10.0, 20.0],
            :flag => [true, false]
        ))
        
        @test length(combo_collection) == 4  # 2 × 2
        
        # Check all combinations exist
        combinations = [(s.overrides[:x], s.overrides[:flag]) for s in combo_collection.scenarios]
        expected_combinations = [(10.0, true), (10.0, false), (20.0, true), (20.0, false)]
        @test Set(combinations) == Set(expected_combinations)
        
        # Test empty grid
        empty_collection = create_scenario_grid("empty", data, Dict{Symbol, Vector}())
        @test length(empty_collection) == 1
        @test empty_collection[1].name == "empty_original"
    end
    
    @testset "Scenario Collections" begin
        # Test collection interface
        collection = create_scenario_grid("test", data, Dict(:x => [1.0, 2.0, 3.0]))
        
        # Test indexing
        @test collection[1] isa DataScenario
        @test collection[2].overrides[:x] == 2.0
        
        # Test iteration
        count = 0
        for scenario in collection
            count += 1
            @test scenario isa DataScenario
        end
        @test count == 3
        
        # Test get_scenario_by_name
        scenario = get_scenario_by_name(collection, "test_x_2.0")
        @test scenario.overrides[:x] == 2.0
        
        # Test error for missing scenario
        @test_throws ErrorException get_scenario_by_name(collection, "nonexistent")
    end
    
    @testset "Integration with modelrow!" begin
        # Test basic scenario evaluation
        scenario = create_scenario("eval_test", data; x = 100.0)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        result = modelrow!(row_vec, compiled, scenario, 1)
        @test result === row_vec
        @test all(isfinite.(row_vec))
        
        # Test that override actually changes results
        normal_vec = Vector{Float64}(undef, length(compiled))
        modelrow!(normal_vec, compiled, data, 1)
        @test row_vec != normal_vec  # Should be different due to x override
        
        # Test multi-scenario evaluation
        scenarios = [
            create_scenario("s1", data; x = 1.0),
            create_scenario("s2", data; x = 2.0),
            create_scenario("s3", data; x = 3.0)
        ]
        
        matrix = Matrix{Float64}(undef, 3, length(compiled))
        modelrow_scenarios!(matrix, compiled, scenarios, 1)
        
        @test all(isfinite.(matrix))
        @test matrix[1, :] != matrix[2, :]  # Different x values
        @test matrix[2, :] != matrix[3, :]
        
        # Test that results are consistent
        single_vec = Vector{Float64}(undef, length(compiled))
        modelrow!(single_vec, compiled, scenarios[2], 1)
        @test matrix[2, :] == single_vec
    end
    
    @testset "Complex Scenario Examples" begin
        # Test scenario with categorical override
        cat_scenario = create_scenario("cat_test", data; group = "B")
        row_vec = Vector{Float64}(undef, length(compiled))
        modelrow!(row_vec, compiled, cat_scenario, 1)
        @test all(isfinite.(row_vec))
        
        # Test scenario with multiple types
        mixed_scenario = create_scenario("mixed", data; 
            x = mean([1.0, 2.0, 3.0, 4.0, 5.0]),
            flag = false,
            group = "A"
        )
        modelrow!(row_vec, compiled, mixed_scenario, 1)
        @test all(isfinite.(row_vec))
        
        # Test that overrides are applied correctly
        @test mixed_scenario.data.x[1] == 3.0  # mean of x
        @test mixed_scenario.data.flag[1] == false
        @test string(mixed_scenario.data.group[1]) == "A"
    end
    
    @testset "Memory Efficiency" begin
        # Test that OverrideVector doesn't allocate large arrays
        large_override = OverrideVector(42.0, 1_000_000)
        @test length(large_override) == 1_000_000
        @test large_override[1] == 42.0
        @test large_override[1_000_000] == 42.0
        
        # Test that scenario creation is efficient
        large_data = Tables.columntable(DataFrame(
            x = randn(1_000_000),
            y = randn(1_000_000)
        ))
        
        scenario = create_scenario("large", large_data; x = 5.0)
        @test scenario.data.x[1] == 5.0
        @test scenario.data.x[1_000_000] == 5.0
        @test scenario.data.y === large_data.y  # No copy
    end
    
end
