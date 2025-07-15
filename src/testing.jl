# testing.jl

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
