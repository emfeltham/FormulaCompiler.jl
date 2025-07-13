# testing_suite.jl

###############################################################################
# Comprehensive Testing Suite
###############################################################################

"""
    test_complete_pipeline()

Test the complete three-phase pipeline on various formula types.
"""
function test_complete_pipeline()
    println("=== Testing Complete Three-Phase Pipeline ===")
    
    # Create comprehensive test data
    Random.seed!(42)
    n = 100
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        z = abs.(randn(n)) .+ 0.1,  # Positive for log
        w = randn(n),
        group = categorical(rand(["A", "B", "C"], n)),
        binary = categorical(rand(["Yes", "No"], n))
    )
    
    data = Tables.columntable(df)
    
    # Test cases from simple to complex
    test_formulas = [
        (@formula(y ~ 1), "Intercept only"),
        (@formula(y ~ x), "Simple continuous"),
        (@formula(y ~ group), "Simple categorical"),
        (@formula(y ~ x + group), "Mixed terms"),
        (@formula(y ~ x^2), "Power function"),
        (@formula(y ~ log(z)), "Log function"),
        (@formula(y ~ x + x^2 + log(z)), "Multiple functions"),
        (@formula(y ~ x * group), "Simple interaction"),
        (@formula(y ~ x^2 * log(z)), "Complex function interaction"),
        (@formula(y ~ x + x^2 + log(z) + group + w + x*group), "Kitchen sink"),
        (@formula(y ~ x*z*group), "Three-way interaction"),
        (@formula(y ~ (x>0) + log(z)*x), "Boolean and function interaction")
    ]
    
    results = []
    successful = 0
    
    i = 8
    (i, (formula, description)) = collect(enumerate(test_formulas))[i]
    for (i, (formula, description)) in enumerate(test_formulas)
        println("\n" * "="^60)
        println("Test $i: $description")
        println("Formula: $formula")
        println("="^60)
        
        try
            # Build model
            model = lm(formula, df)
            
            # Test complete pipeline
            compilation_time, avg_time, avg_allocs, correctness = test_compilation_performance(
                model, data, n_trials=100
            )
            
            # Record results
            push!(results, (
                description = description,
                formula = string(formula),
                compilation_time = compilation_time,
                avg_time = avg_time,
                avg_allocs = avg_allocs,
                correctness = correctness,
                success = true
            ))
            
            if correctness && avg_allocs < 0.1
                successful += 1
                println("✅ PASSED: Correct and zero-allocation")
            else
                println("⚠️  PARTIAL: Some issues detected")
            end
            
        catch e
            println("❌ FAILED: $e")
            @error "Test $i failed" exception=(e, catch_backtrace())
            
            push!(results, (
                description = description,
                formula = string(formula),
                compilation_time = NaN,
                avg_time = NaN,
                avg_allocs = NaN,
                correctness = false,
                success = false,
                error = string(e)
            ))
        end
    end
    
    # Summary report
    println("\n" * "="^60)
    println("FINAL SUMMARY")
    println("="^60)
    println("Successful tests: $successful / $(length(test_formulas))")
    
    println("\nPerformance Summary:")
    for result in results
        if result.success
            println("  $(result.description): $(round(result.avg_time, digits=1)) ns, $(round(result.avg_allocs, digits=3)) allocs")
        else
            println("  $(result.description): FAILED")
        end
    end
    
    if successful == length(test_formulas)
        println("\n🎉 ALL TESTS PASSED! Three-phase pipeline working perfectly!")
    elseif successful > length(test_formulas) * 0.8
        println("\n✅ Most tests passed, minor issues to resolve")
    else
        println("\n⚠️  Major issues detected, needs debugging")
    end
    
    return results
end