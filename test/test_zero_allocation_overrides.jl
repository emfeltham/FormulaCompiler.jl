# test_zero_allocation_overrides.jl - Zero allocation validation for Part 0 override improvements
#
# Validates that modelrow! with scenarios achieves zero allocations after warmup
# per MARGINS_COMPUTE_REWRITE_PLAN.md Part 0 requirements

using Test
using FormulaCompiler
using GLM, DataFrames, Tables
using BenchmarkTools
using CategoricalArrays
using FormulaCompiler: NumericCounterfactualVector, BoolCounterfactualVector, CategoricalCounterfactualVector, modelrow!

@testset "Zero Allocation Override Performance" begin
    
    # Setup test data
    n = 100
    df = DataFrame(
        y = randn(n),
        x1 = randn(n),
        x2 = randn(n),
        age = rand(18:65, n),
        treated = rand(Bool, n),
        group = categorical(rand(["A", "B", "C"], n), levels=["A", "B", "C"]),
        region = categorical(rand(["North", "South"], n), levels=["North", "South"])
    )
    
    data_nt = Tables.columntable(df)
    model = lm(@formula(y ~ x1 + x2 + treated + group), df)
    compiled = compile_formula(model, data_nt)
    
    @testset "Continuous Variable Overrides" begin
        cf_x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 2.0)
        cf_x2 = NumericCounterfactualVector{Float64}(data_nt.x2, 1, -1.5)
        cf_data = merge(data_nt, (x1 = cf_x1, x2 = cf_x2))
        output = Vector{Float64}(undef, length(compiled))

        # Warmup
        modelrow!(output, compiled, cf_data, 1)

        # Benchmark
        result = @benchmark modelrow!($output, $compiled, $cf_data, 1) samples=200 seconds=2
        
        println("Continuous override benchmark:")
        println("  Memory: $(result.memory) bytes")
        println("  Allocations: $(result.allocs)")
        println("  Min time: $(minimum(result.times) / 1000) μs")
        
        @test result.memory == 0
        @test result.allocs == 0
    end
    
    @testset "Boolean Variable Overrides" begin
        cf_treated = BoolCounterfactualVector(data_nt.treated, 1, true)
        cf_data = merge(data_nt, (treated = cf_treated,))
        output = Vector{Float64}(undef, length(compiled))

        # Warmup
        modelrow!(output, compiled, cf_data, 1)

        # Benchmark
        result = @benchmark modelrow!($output, $compiled, $cf_data, 1) samples=200 seconds=2
        
        println("Boolean override benchmark:")
        println("  Memory: $(result.memory) bytes")
        println("  Allocations: $(result.allocs)")
        println("  Min time: $(minimum(result.times) / 1000) μs")
        
        @test result.memory == 0
        @test result.allocs == 0
    end
    
    @testset "Categorical Variable Overrides" begin
        # Get appropriate categorical values
        group_b = CategoricalValue("B", data_nt.group)
        region_south = CategoricalValue("South", data_nt.region)
        cf_group = CategoricalCounterfactualVector(data_nt.group, 1, group_b)
        cf_region = CategoricalCounterfactualVector(data_nt.region, 1, region_south)
        cf_data = merge(data_nt, (group = cf_group, region = cf_region))
        output = Vector{Float64}(undef, length(compiled))

        # Warmup
        modelrow!(output, compiled, cf_data, 1)

        # Benchmark
        result = @benchmark modelrow!($output, $compiled, $cf_data, 1) samples=200 seconds=2
        
        println("Categorical override benchmark:")
        println("  Memory: $(result.memory) bytes")
        println("  Allocations: $(result.allocs)")
        println("  Min time: $(minimum(result.times) / 1000) μs")
        
        @test result.memory == 0
        @test result.allocs == 0
    end
    
    @testset "Mixed Variable Overrides" begin
        cf_x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 1.5)
        cf_x2 = NumericCounterfactualVector{Float64}(data_nt.x2, 1, -0.8)
        cf_age = NumericCounterfactualVector{Int64}(data_nt.age, 1, 40)
        cf_treated = BoolCounterfactualVector(data_nt.treated, 1, false)
        group_c = CategoricalValue("C", data_nt.group)
        region_north = CategoricalValue("North", data_nt.region)
        cf_group = CategoricalCounterfactualVector(data_nt.group, 1, group_c)
        cf_region = CategoricalCounterfactualVector(data_nt.region, 1, region_north)
        cf_data = merge(data_nt, (x1 = cf_x1, x2 = cf_x2, age = cf_age, treated = cf_treated, group = cf_group, region = cf_region))
        output = Vector{Float64}(undef, length(compiled))

        # Warmup
        modelrow!(output, compiled, cf_data, 1)

        # Benchmark
        result = @benchmark modelrow!($output, $compiled, $cf_data, 1) samples=200 seconds=2
        
        println("Mixed override benchmark:")
        println("  Memory: $(result.memory) bytes")
        println("  Allocations: $(result.allocs)")
        println("  Min time: $(minimum(result.times) / 1000) μs")
        
        @test result.memory == 0
        @test result.allocs == 0
    end
    
    @testset "Multiple Rows Zero Allocation" begin
        cf_x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 2.5)
        cf_treated = BoolCounterfactualVector(data_nt.treated, 1, true)
        group_a = CategoricalValue("A", data_nt.group)
        cf_group = CategoricalCounterfactualVector(data_nt.group, 1, group_a)
        cf_data = merge(data_nt, (x1 = cf_x1, treated = cf_treated, group = cf_group))
        output = Vector{Float64}(undef, length(compiled))

        # Test different rows
        test_rows = [1, 10, 25, 50, 75, 100]

        for row_idx in test_rows
            # Update counterfactual to apply to this row
            FormulaCompiler.update_counterfactual_row!(cf_x1, row_idx)
            FormulaCompiler.update_counterfactual_row!(cf_treated, row_idx)
            FormulaCompiler.update_counterfactual_row!(cf_group, row_idx)

            # Warmup
            modelrow!(output, compiled, cf_data, row_idx)

            # Benchmark
            result = @benchmark modelrow!($output, $compiled, $cf_data, $row_idx) samples=50 seconds=1
            
            @test result.memory == 0
            @test result.allocs == 0
        end
        
        println("✓ All $(length(test_rows)) test rows achieved zero allocation")
    end
    
    @testset "Comparison: Normal vs Override Performance" begin
        # Benchmark normal modelrow! without overrides
        output = Vector{Float64}(undef, length(compiled))
        
        # Warmup
        modelrow!(output, compiled, data_nt, 1)
        
        # Benchmark normal
        normal_result = @benchmark modelrow!($output, $compiled, $data_nt, 1) samples=200 seconds=2
        
        # Benchmark with override
        cf_x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 2.0)
        cf_treated = BoolCounterfactualVector(data_nt.treated, 1, true)
        cf_data = merge(data_nt, (x1 = cf_x1, treated = cf_treated))
        modelrow!(output, compiled, cf_data, 1)  # Warmup

        override_result = @benchmark modelrow!($output, $compiled, $cf_data, 1) samples=200 seconds=2
        
        println("Performance comparison:")
        println("  Normal - Memory: $(normal_result.memory) bytes, Allocs: $(normal_result.allocs)")
        println("  Override - Memory: $(override_result.memory) bytes, Allocs: $(override_result.allocs)")
        println("  Normal - Min time: $(minimum(normal_result.times) / 1000) μs")
        println("  Override - Min time: $(minimum(override_result.times) / 1000) μs")
        
        @test normal_result.memory == 0
        @test override_result.memory == 0
        @test normal_result.allocs == 0
        @test override_result.allocs == 0
        
        # Performance should be comparable (within 2x)
        normal_time = minimum(normal_result.times)
        override_time = minimum(override_result.times)
        performance_ratio = override_time / normal_time
        
        @test performance_ratio < 2.0
        @debug "Override performance should be within 2x of normal (got $(performance_ratio)x)"
        println("✓ Override performance ratio: $(round(performance_ratio, digits=2))x")
    end
    
    @testset "Complex Model Zero Allocation" begin
        # Test with interactions and transforms
        complex_df = DataFrame(
            y = randn(50),
            x1 = randn(50),
            x2 = randn(50),
            age = rand(25:60, 50),
            treated = rand(Bool, 50),
            group = categorical(rand(["A", "B"], 50), levels=["A", "B"])
        )
        complex_data = Tables.columntable(complex_df)
        
        # Complex model with interactions and transforms
        complex_model = lm(@formula(y ~ x1 + log(age) + treated * x2 + group + x1^2), complex_df)
        complex_compiled = compile_formula(complex_model, complex_data)
        
        cf_x1 = NumericCounterfactualVector{Float64}(complex_data.x1, 1, 1.5)
        cf_age = NumericCounterfactualVector{Int64}(complex_data.age, 1, 35)
        cf_treated = BoolCounterfactualVector(complex_data.treated, 1, true)
        cf_x2 = NumericCounterfactualVector{Float64}(complex_data.x2, 1, -0.5)
        group_b = CategoricalValue("B", complex_data.group)
        cf_group = CategoricalCounterfactualVector(complex_data.group, 1, group_b)
        cf_data = merge(complex_data, (x1 = cf_x1, age = cf_age, treated = cf_treated, x2 = cf_x2, group = cf_group))
        output = Vector{Float64}(undef, length(complex_compiled))

        # Warmup
        modelrow!(output, complex_compiled, cf_data, 1)

        # Benchmark
        result = @benchmark modelrow!($output, $complex_compiled, $cf_data, 1) samples=200 seconds=2
        
        println("Complex model override benchmark:")
        println("  Memory: $(result.memory) bytes")
        println("  Allocations: $(result.allocs)")
        println("  Min time: $(minimum(result.times) / 1000) μs")
        
        @test result.memory == 0
        @test result.allocs == 0
        @test all(isfinite.(output))
    end
    
end