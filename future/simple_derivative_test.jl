# simple_derivative_test.jl
# Quick test to verify the core mathematical fixes work

using FormulaCompiler
using DataFrames, GLM, Tables

function test_core_derivative_fixes()
    println("🧪 Testing Core Derivative Fixes")
    println("=" ^ 40)
    
    # Clear caches
    FormulaCompiler.clear_derivative_cache!()
    empty!(FormulaCompiler.FORMULA_CACHE)
    
    # Test 1: Log derivative fix
    println("1. Testing log derivative: ∂log(z)/∂z = 1/z")
    
    df = DataFrame(y = [1.0, 2.0, 3.0], z = [1.0, 2.0, 4.0])
    model = lm(@formula(y ~ log(z)), df)
    compiled = compile_formula(model)
    dz_compiled = compile_derivative_formula(compiled, :z)
    
    data = Tables.columntable(df)
    deriv_vec = Vector{Float64}(undef, length(dz_compiled))
    
    success = true
    for i in 1:nrow(df)
        modelrow!(deriv_vec, dz_compiled, data, i)
        expected = 1.0 / df.z[i]
        actual = deriv_vec[2]  # Position after intercept
        error = abs(actual - expected)
        
        println("   Row $i: z=$(df.z[i]), got $actual, expected $expected, error=$error")
        
        if error > 1e-12
            success = false
        end
    end
    
    if success
        println("   ✅ Log derivative test PASSED")
    else
        println("   ❌ Log derivative test FAILED")
    end
    
    # Test 2: Power derivative fix
    println("\n2. Testing power derivative: ∂(x²)/∂x = 2x")
    
    df2 = DataFrame(y = [1.0, 4.0, 9.0], x = [1.0, 2.0, 3.0])
    model2 = lm(@formula(y ~ x^2), df2)
    compiled2 = compile_formula(model2)
    dx_compiled2 = compile_derivative_formula(compiled2, :x)
    
    data2 = Tables.columntable(df2)
    deriv_vec2 = Vector{Float64}(undef, length(dx_compiled2))
    
    success2 = true
    for i in 1:nrow(df2)
        modelrow!(deriv_vec2, dx_compiled2, data2, i)
        expected = 2.0 * df2.x[i]
        actual = deriv_vec2[2]  # Position after intercept
        error = abs(actual - expected)
        
        println("   Row $i: x=$(df2.x[i]), got $actual, expected $expected, error=$error")
        
        if error > 1e-12
            success2 = false
        end
    end
    
    if success2
        println("   ✅ Power derivative test PASSED")
    else
        println("   ❌ Power derivative test FAILED")
    end
    
    # Test 3: Zero allocations
    println("\n3. Testing zero allocations")
    
    # Warm up
    modelrow!(deriv_vec, dz_compiled, data, 1)
    
    # Test allocations
    allocs = @allocated modelrow!(deriv_vec, dz_compiled, data, 1)
    println("   Allocations: $allocs bytes")
    
    if allocs == 0
        println("   ✅ Zero allocations achieved")
    else
        println("   ⚠️  Non-zero allocations")
    end
    
    # Summary
    overall_success = success && success2 && (allocs == 0)
    println("\n" * "^" * "=" * "^" * "40")
    if overall_success
        println("🎉 ALL CORE TESTS PASSED!")
        println("✅ Mathematical correctness verified")
        println("✅ Zero allocations maintained")
    else
        println("⚠️  Some tests failed - check output above")
    end
    
    return overall_success
end

# Run the test
test_core_derivative_fixes()