# test/test_helpers.jl
# Helper functions for testing

"""
    create_test_data(n::Int=100) -> DataFrame

Create standardized test data for consistent testing.
"""
function create_test_data(n::Int=100)
    Random.seed!(42)
    return DataFrame(
        x = randn(n),
        y = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n)),
        flag = rand([true, false], n),
        cat2 = categorical(rand(["X", "Y"], n)),
        cat3 = categorical(rand(["P", "Q", "R"], n)),
        numeric_int = rand(1:10, n),
        numeric_float32 = Float32.(randn(n))
    )
end

"""
    test_model_correctness(model, df; rtol=1e-12)

Test that a compiled model produces correct results compared to modelmatrix.
"""
function test_model_correctness(model, df; rtol=1e-12)
    compiled = compile_formula(model)
    data = Tables.columntable(df)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Test multiple rows
    mm = modelmatrix(model)
    for i in 1:min(10, nrow(df))
        compiled(row_vec, data, i)
        expected = mm[i, :]
        if !isapprox(row_vec, expected, rtol=rtol)
            @error "Model correctness test failed" i row_vec expected
            return false
        end
    end
    return true
end

"""
    test_zero_allocations(compiled, data, row_vec)

Test that evaluation has zero allocations.
"""
function test_zero_allocations(compiled, data, row_vec)
    allocs = @allocated compiled(row_vec, data, 1)
    return allocs == 0
end

"""
    test_performance_benchmarks(compiled, data, row_vec)

Test performance benchmarks and return timing statistics.
"""
function test_performance_benchmarks(compiled, data, row_vec)
    # Single evaluation time
    eval_time = @elapsed compiled(row_vec, data, 1)
    
    # Benchmark with multiple runs
    benchmark_result = @benchmark $compiled($row_vec, $data, 1)
    
    return (
        single_time = eval_time,
        median_time = median(benchmark_result.times),
        mean_time = mean(benchmark_result.times),
        std_time = std(benchmark_result.times),
        min_time = minimum(benchmark_result.times),
        max_time = maximum(benchmark_result.times)
    )
end

"""
    comprehensive_model_test(formula, df; test_performance=true, test_derivatives=false)

Run comprehensive tests on a model formula.
"""
function comprehensive_model_test(formula, df; test_performance=true, test_derivatives=false)
    results = Dict{Symbol, Any}()
    
    try
        # Basic model fitting and compilation
        model = lm(formula, df)
        compiled = compile_formula(model)
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        # Test correctness
        results[:correctness] = test_model_correctness(model, df)
        
        # Test zero allocations
        results[:zero_allocations] = test_zero_allocations(compiled, data, row_vec)
        
        # Test performance if requested
        if test_performance
            results[:performance] = test_performance_benchmarks(compiled, data, row_vec)
        end
        
        # Test evaluator tree access
        results[:evaluator_access] = has_evaluator_access(compiled)
        results[:node_count] = count_evaluator_nodes(compiled)
        results[:variable_dependencies] = get_variable_dependencies(compiled)
        
        # Test derivatives if requested
        if test_derivatives
            try
                root_eval = extract_root_evaluator(compiled)
                # Test derivative for first continuous variable
                continuous_vars = filter(v -> v in names(df) && eltype(df[!, v]) <: Number, results[:variable_dependencies])
                if !isempty(continuous_vars)
                    focal_var = continuous_vars[1]
                    deriv_eval = compute_derivative_evaluator(root_eval, focal_var)
                    results[:derivative_test] = validate_derivative_evaluator(deriv_eval, focal_var, data)
                end
            catch e
                results[:derivative_test] = false
                results[:derivative_error] = e
            end
        end
        
        # Test modelrow interfaces
        results[:modelrow_interfaces] = test_modelrow_interfaces(model, data)
        
        # Overall success
        results[:success] = true
        
    catch e
        results[:success] = false
        results[:error] = e
    end
    
    return results
end

"""
    test_modelrow_interfaces(model, data)

Test all modelrow interfaces for consistency.
"""
function test_modelrow_interfaces(model, data)
    try
        # Pre-compiled interface
        compiled = compile_formula(model)
        row_vec1 = Vector{Float64}(undef, length(compiled))
        modelrow!(row_vec1, compiled, data, 1)
        
        # Cached interface
        row_vec2 = Vector{Float64}(undef, length(compiled))
        modelrow!(row_vec2, model, data, 1; cache=true)
        
        # Non-cached interface
        row_vec3 = Vector{Float64}(undef, length(compiled))
        modelrow!(row_vec3, model, data, 1; cache=false)
        
        # Allocating interface
        row_vec4 = modelrow(model, data, 1)
        
        # Check consistency
        return (row_vec1 == row_vec2 == row_vec3 == row_vec4)
        
    catch e
        return false
    end
end

"""
    run_formula_test_suite(formulas, df; verbose=false)

Run a comprehensive test suite on multiple formulas.
"""
function run_formula_test_suite(formulas, df; verbose=false)
    results = Dict{String, Any}()
    
    for (i, formula) in enumerate(formulas)
        formula_str = string(formula)
        if verbose
            println("Testing formula $i: $formula_str")
        end
        
        test_result = comprehensive_model_test(formula, df)
        results[formula_str] = test_result
        
        if verbose
            if test_result[:success]
                println("  ✅ PASSED")
            else
                println("  ❌ FAILED: $(test_result[:error])")
            end
        end
    end
    
    return results
end

"""
    summarize_test_results(results)

Summarize test results across multiple formulas.
"""
function summarize_test_results(results)
    total_tests = length(results)
    successful_tests = sum(r[:success] for r in values(results))
    
    println("Test Summary:")
    println("=============")
    println("Total formulas tested: $total_tests")
    println("Successful: $successful_tests")
    println("Failed: $(total_tests - successful_tests)")
    println("Success rate: $(round(successful_tests/total_tests*100, digits=1))%")
    
    if successful_tests > 0
        # Performance summary
        performance_data = [r[:performance] for r in values(results) if r[:success] && haskey(r, :performance)]
        if !isempty(performance_data)
            median_times = [p[:median_time] for p in performance_data]
            println("\nPerformance Summary:")
            println("Median evaluation time: $(round(median(median_times), digits=1)) ns")
            println("Range: $(round(minimum(median_times), digits=1)) - $(round(maximum(median_times), digits=1)) ns")
        end
        
        # Zero allocation summary
        zero_alloc_count = sum(r[:zero_allocations] for r in values(results) if r[:success])
        println("\nZero allocation tests: $zero_alloc_count / $successful_tests")
        
        # Correctness summary
        correct_count = sum(r[:correctness] for r in values(results) if r[:success])
        println("Correctness tests: $correct_count / $successful_tests")
    end
    
    # Failed tests
    failed_tests = [name for (name, result) in results if !result[:success]]
    if !isempty(failed_tests)
        println("\nFailed tests:")
        for test in failed_tests
            println("  - $test")
        end
    end
end

"""
    create_standard_test_formulas() -> Vector{Expr}

Create a standard set of test formulas covering common patterns.
"""
function create_standard_test_formulas()
    return [
        # Basic formulas
        @formula(y ~ x),
        @formula(y ~ x + z),
        @formula(y ~ x * z),
        
        # Categorical formulas
        @formula(y ~ group),
        @formula(y ~ x + group),
        @formula(y ~ x * group),
        
        # Function formulas
        @formula(y ~ log(z)),
        @formula(y ~ x + log(z)),
        @formula(y ~ x * log(z)),
        
        # Complex interactions
        @formula(y ~ x * z * group),
        @formula(y ~ x * group + z * group),
        @formula(y ~ (x + z) * group),
        
        # Boolean expressions
        @formula(y ~ (x > 0)),
        @formula(y ~ (x > 0) * group),
        @formula(y ~ flag),
        @formula(y ~ flag * group),
        
        # Polynomial terms
        @formula(y ~ x + x^2),
        @formula(y ~ x * x^2),
        @formula(y ~ x^2 * group),
        
        # Mixed function and categorical
        @formula(y ~ log(z) * group),
        @formula(y ~ sqrt(abs(x)) * group),
        @formula(y ~ sin(x) * group),
        
        # Multi-categorical
        @formula(y ~ group * cat2),
        @formula(y ~ group * cat2 * cat3),
        @formula(y ~ x * group * cat2),
        
        # Edge cases
        @formula(y ~ 1),  # Intercept only
        @formula(y ~ 0 + x),  # No intercept
        @formula(y ~ x - 1),  # Alternative no intercept syntax
    ]
end

"""
    benchmark_against_modelmatrix(model, df; n_trials=1000)

Benchmark compiled formula against modelmatrix for performance comparison.
"""
function benchmark_against_modelmatrix(model, df; n_trials=1000)
    compiled = compile_formula(model)
    data = Tables.columntable(df)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Benchmark modelmatrix approach
    mm_times = Float64[]
    for _ in 1:n_trials
        row_idx = rand(1:nrow(df))
        time = @elapsed begin
            mm = modelmatrix(model)
            result = mm[row_idx, :]
        end
        push!(mm_times, time)
    end
    
    # Benchmark compiled approach
    compiled_times = Float64[]
    for _ in 1:n_trials
        row_idx = rand(1:nrow(df))
        time = @elapsed compiled(row_vec, data, row_idx)
        push!(compiled_times, time)
    end
    
    return (
        modelmatrix_median = median(mm_times),
        modelmatrix_mean = mean(mm_times),
        compiled_median = median(compiled_times),
        compiled_mean = mean(compiled_times),
        speedup_median = median(mm_times) / median(compiled_times),
        speedup_mean = mean(mm_times) / mean(compiled_times)
    )
end

"""
    test_compilation_robustness(formulas, df; max_errors=5)

Test robustness of compilation across many formulas.
"""
function test_compilation_robustness(formulas, df; max_errors=5)
    errors = []
    successful = 0
    
    for formula in formulas
        try
            model = lm(formula, df)
            compiled = compile_formula(model)
            
            # Basic smoke test
            data = Tables.columntable(df)
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, data, 1)
            
            successful += 1
        catch e
            push!(errors, (formula=formula, error=e))
            if length(errors) >= max_errors
                break
            end
        end
    end
    
    return (
        total_tested = successful + length(errors),
        successful = successful,
        errors = errors,
        success_rate = successful / (successful + length(errors))
    )
end

"""
    stress_test_performance(model, df; n_evaluations=100_000)

Stress test performance with many evaluations.
"""
function stress_test_performance(model, df; n_evaluations=100_000)
    compiled = compile_formula(model)
    data = Tables.columntable(df)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Warm up
    for i in 1:100
        compiled(row_vec, data, (i-1) % nrow(df) + 1)
    end
    
    # Stress test
    start_time = time_ns()
    total_allocations = 0
    
    for i in 1:n_evaluations
        row_idx = (i-1) % nrow(df) + 1
        allocs = @allocated compiled(row_vec, data, row_idx)
        total_allocations += allocs
    end
    
    end_time = time_ns()
    total_time = (end_time - start_time) / 1e9  # Convert to seconds
    
    return (
        total_evaluations = n_evaluations,
        total_time = total_time,
        avg_time_per_eval = total_time / n_evaluations,
        evaluations_per_second = n_evaluations / total_time,
        total_allocations = total_allocations,
        avg_allocations = total_allocations / n_evaluations
    )
end
