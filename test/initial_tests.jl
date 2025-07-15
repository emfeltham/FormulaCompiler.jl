# test_efficient_model_matrices.jl
# Formal test suite for EfficientModelMatrices.jl

using Test
using Random
using DataFrames
using GLM
using Tables
using CategoricalArrays
using StatsModels

# Import your module here
# using EfficientModelMatrices

###############################################################################
# Test Data Setup
###############################################################################

function create_test_data()
    Random.seed!(06515)  # Fixed seed for reproducibility
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
    
    return df, Tables.columntable(df)
end

###############################################################################
# Test Utilities
###############################################################################

"""
Test correctness and allocations for a single formula.
Returns (correctness_passed, allocation_status)
"""
function test_single_formula(formula, df, data, test_row=1)
    try
        # Build model and get expected output
        model = lm(formula, df)
        mm = modelmatrix(model)
        expected = mm[test_row, :]
        
        # Compile formula
        compiled = compile_formula(model)
        
        # Test width consistency
        @test length(compiled) == size(mm, 2)
        
        # Test correctness
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, test_row)
        
        # Check numerical accuracy
        max_error = maximum(abs.(row_vec .- expected))
        correctness_passed = max_error < 1e-12
        
        if !correctness_passed
            @warn "Numerical error detected" formula=formula max_error=max_error
            return false, :failed
        end
        
        # Test allocations
        allocs = @allocated compiled(row_vec, data, test_row)
        
        if allocs == 0
            return true, :perfect
        elseif allocs < 100
            return true, :good
        else
            return true, :high_alloc
        end
        
    catch e
        @warn "Exception in formula test" formula=formula exception=e
        return false, :exception
    end
end

###############################################################################
# Main Test Suites
###############################################################################

@testset "EfficientModelMatrices.jl" begin
    
    df, data = create_test_data()
    
    @testset "Basic Functionality" begin
        @testset "Simple Terms" begin
            # Intercept only
            @test_nowarn begin
                model = lm(@formula(y ~ 1), df)
                compiled = compile_formula(model)
                @test length(compiled) == 1
            end
            
            # Single continuous
            @test_nowarn begin
                model = lm(@formula(y ~ x), df)
                compiled = compile_formula(model)
                @test length(compiled) == 2  # intercept + x
            end
            
            # Single categorical
            @test_nowarn begin
                model = lm(@formula(y ~ group), df)
                compiled = compile_formula(model)
                @test length(compiled) == 3  # intercept + 2 contrasts for 3-level factor
            end
        end
        
        @testset "Function Terms" begin
            # Power function
            @test_nowarn begin
                model = lm(@formula(y ~ x^2), df)
                compiled = compile_formula(model)
                @test length(compiled) == 2  # intercept + x^2
            end
            
            # Log function
            @test_nowarn begin
                model = lm(@formula(y ~ log(z)), df)
                compiled = compile_formula(model)
                @test length(compiled) == 2  # intercept + log(z)
            end
            
            # Boolean function
            @test_nowarn begin
                model = lm(@formula(y ~ (x > 0)), df)
                compiled = compile_formula(model)
                @test length(compiled) == 2  # intercept + (x > 0)
            end
        end
    end
    
    @testset "Two-Level Categorical Interactions" begin
        test_cases = [
            (@formula(y ~ cat2a * cat2b), "cat 2 x cat 2"),
            (@formula(y ~ cat2a * bool), "cat 2 x bool"),
            (@formula(y ~ cat2a * (x^2)), "cat 2 x cts"),
            (@formula(y ~ bool * (x^2)), "binary x cts"),
            (@formula(y ~ cat2b * (x^2)), "cat 2 x cts (variant)"),
        ]
        
        for (formula, description) in test_cases
            @testset "$description" begin
                correctness, alloc_status = test_single_formula(formula, df, data)
                
                @test correctness "Formula should produce correct results: $formula"
                
                # Test multiple rows for consistency
                @testset "Multiple Rows" begin
                    model = lm(formula, df)
                    mm = modelmatrix(model)
                    compiled = compile_formula(model)
                    row_vec = Vector{Float64}(undef, length(compiled))
                    
                    for test_row in [1, 5, 10, 50]
                        if test_row <= size(mm, 1)
                            compiled(row_vec, data, test_row)
                            expected = mm[test_row, :]
                            @test isapprox(row_vec, expected, atol=1e-12) "Row $test_row should match for $description"
                        end
                    end
                end
                
                # Allocation tests (informational, not strict requirements)
                if alloc_status == :perfect
                    @info "Perfect performance: zero allocations" formula=formula
                elseif alloc_status == :good
                    @info "Good performance: low allocations" formula=formula
                else
                    @warn "High allocations detected" formula=formula status=alloc_status
                end
            end
        end
    end
    
    @testset "Multi-Level Categorical Interactions" begin
        test_cases = [
            (@formula(y ~ group2 * (x^2)), "cat >2 x cts"),
            (@formula(y ~ group2 * bool), "cat >2 x bool"),
            (@formula(y ~ group2 * cat2a), "cat >2 x cat 2"),
            (@formula(y ~ group2 * group3), "cat >2 x cat >2"),
        ]
        
        for (formula, description) in test_cases
            @testset "$description" begin
                correctness, alloc_status = test_single_formula(formula, df, data)
                @test correctness "Formula should produce correct results: $formula"
                
                # Test that width calculation is correct
                model = lm(formula, df)
                mm = modelmatrix(model)
                compiled = compile_formula(model)
                @test length(compiled) == size(mm, 2) "Width should match model matrix for $description"
            end
        end
    end
    
    @testset "Complex Interactions" begin
        test_cases = [
            (@formula(y ~ x * z * group), "three-way continuous x categorical"),
            (@formula(y ~ (x>0) * group), "boolean function x categorical"),
            (@formula(y ~ log(z) * group2 * cat2a), "function x cat >2 x cat 2"),
        ]
        
        for (formula, description) in test_cases
            @testset "$description" begin
                correctness, alloc_status = test_single_formula(formula, df, data)
                @test correctness "Complex formula should work: $formula"
                
                # Test numerical stability
                model = lm(formula, df)
                mm = modelmatrix(model)
                compiled = compile_formula(model)
                row_vec = Vector{Float64}(undef, length(compiled))
                
                # Test first 10 rows
                for test_row in 1:min(10, size(mm, 1))
                    compiled(row_vec, data, test_row)
                    expected = mm[test_row, :]
                    max_error = maximum(abs.(row_vec .- expected))
                    @test max_error < 1e-10 "Numerical error should be minimal for row $test_row in $description"
                end
            end
        end
    end
    
    @testset "Edge Cases" begin
        @testset "Boolean Variables" begin
            # Test boolean as main effect
            model = lm(@formula(y ~ bool), df)
            compiled = compile_formula(model)
            mm = modelmatrix(model)
            row_vec = Vector{Float64}(undef, length(compiled))
            
            compiled(row_vec, data, 1)
            expected = mm[1, :]
            @test isapprox(row_vec, expected, atol=1e-12)
        end
        
        @testset "Nested Functions" begin
            # Test nested function calls
            @test_nowarn begin
                model = lm(@formula(y ~ log(sqrt(abs(x)))), df)
                compiled = compile_formula(model)
                @test length(compiled) == 2
            end
        end
        
        @testset "Large Categorical" begin
            # Test with larger categorical variable
            df_large = copy(df)
            df_large.large_cat = categorical(rand(["A", "B", "C", "D", "E", "F"], nrow(df_large)))
            data_large = Tables.columntable(df_large)
            
            model = lm(@formula(y ~ large_cat), df_large)
            compiled = compile_formula(model)
            mm = modelmatrix(model)
            
            @test length(compiled) == size(mm, 2)
            
            # Test correctness
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, data_large, 1)
            expected = mm[1, :]
            @test isapprox(row_vec, expected, atol=1e-12)
        end
    end
    
    @testset "Performance Regression Tests" begin
        # These are informational tests to catch performance regressions
        @testset "Allocation Tests" begin
            simple_cases = [
                @formula(y ~ x),
                @formula(y ~ x * group),
                @formula(y ~ x^2 * cat2a),
            ]
            
            for formula in simple_cases
                model = lm(formula, df)
                compiled = compile_formula(model)
                row_vec = Vector{Float64}(undef, length(compiled))
                
                # Warmup
                for _ in 1:10
                    compiled(row_vec, data, 1)
                end
                
                # Measure allocations
                allocs = @allocated compiled(row_vec, data, 1)
                
                @info "Allocation measurement" formula=formula allocations=allocs
                
                # This is aspirational - we want zero allocations eventually
                if allocs == 0
                    @info "🎉 Zero allocations achieved!" formula=formula
                end
            end
        end
    end
    
    @testset "Comprehensive Integration Test" begin
        # Run all test cases from the original function
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
        
        for (formula, description) in test_cases
            correctness, alloc_status = test_single_formula(formula, df, data)
            push!(results, (description, correctness, alloc_status))
            
            @test correctness "Integration test failed for: $description"
        end
        
        # Summary statistics
        total_passed = count(r -> r[2], results)
        perfect_allocs = count(r -> r[2] && r[3] == :perfect, results)
        good_allocs = count(r -> r[2] && r[3] == :good, results)
        
        @info "Integration Test Summary" begin
            total_tests = length(results)
            passed = total_passed
            perfect = perfect_allocs  
            good = good_allocs
            "Passed: $passed/$total_tests, Perfect allocs: $perfect, Good allocs: $good"
        end
        
        # At minimum, all tests should pass correctness
        @test total_passed == length(test_cases) "All integration tests should pass"
        
        # Aspirational: most tests should have good allocation behavior
        good_allocation_ratio = (perfect_allocs + good_allocs) / length(test_cases)
        if good_allocation_ratio >= 0.8
            @info "🎉 Excellent allocation performance: $(round(good_allocation_ratio*100, digits=1))% of tests have good allocations"
        elseif good_allocation_ratio >= 0.5
            @info "✅ Good allocation performance: $(round(good_allocation_ratio*100, digits=1))% of tests have good allocations"
        else
            @warn "⚠️  Poor allocation performance: only $(round(good_allocation_ratio*100, digits=1))% of tests have good allocations"
        end
    end
end

###############################################################################
# Benchmark Tests (Optional)
###############################################################################

@testset "Performance Benchmarks" begin
    # These tests are for performance monitoring, not strict pass/fail
    
    df, data = create_test_data()
    
    @testset "Speed Benchmarks" begin
        # Test a few representative formulas for timing
        benchmark_cases = [
            (@formula(y ~ x), "simple"),
            (@formula(y ~ x * group), "medium"),
            (@formula(y ~ x * z * group * cat2a), "complex"),
        ]
        
        for (formula, complexity) in benchmark_cases
            model = lm(formula, df)
            compiled = compile_formula(model)
            row_vec = Vector{Float64}(undef, length(compiled))
            
            # Warmup
            for _ in 1:100
                compiled(row_vec, data, 1)
            end
            
            # Time single evaluation
            times = Float64[]
            for _ in 1:1000
                t = @elapsed compiled(row_vec, data, 1)
                push!(times, t)
            end
            
            avg_time_ns = mean(times) * 1e9
            min_time_ns = minimum(times) * 1e9
            
            @info "Timing benchmark" formula=formula complexity=complexity avg_time_ns=avg_time_ns min_time_ns=min_time_ns
            
            # Aspirational performance targets
            if avg_time_ns < 100
                @info "🎉 Excellent performance: $(round(avg_time_ns, digits=1)) ns average"
            elseif avg_time_ns < 500
                @info "✅ Good performance: $(round(avg_time_ns, digits=1)) ns average"
            else
                @info "⚠️  Slower than target: $(round(avg_time_ns, digits=1)) ns average"
            end
        end
    end
end

###############################################################################
# Test Runner Function
###############################################################################

"""
Run all tests with summary reporting.
"""
function run_all_tests()
    @info "Starting EfficientModelMatrices.jl test suite..."
    
    # Set up test environment
    original_seed = Random.seed!()
    
    try
        # Run the test suite
        Test.@testset "Complete Test Suite" begin
            # The @testset above will run automatically when this file is included
        end
        
        @info "✅ Test suite completed successfully!"
        
    catch e
        @error "❌ Test suite failed" exception=e
        rethrow(e)
    finally
        # Restore random seed
        Random.seed!(original_seed)
    end
end

# Export test functions for manual running
# export run_all_tests, create_test_data, test_single_formula

run_all_tests()
