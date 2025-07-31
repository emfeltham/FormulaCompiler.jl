###############################################################################
# COMPREHENSIVE TEST FUNCTION
###############################################################################

"""
    test_comprehensive_fix(formula, df, data)

Test both function and interaction fixes comprehensively using standard function names.
"""
function test_comprehensive_fix(formula, df, data)
    println("="^60)
    println("TESTING COMPREHENSIVE PHASE 2 FIX")
    println("Formula: $formula")
    println("="^60)
    
    # Compile using standard function names (now with fixes applied)
    model = fit(LinearModel, formula, df)
    specialized = compile_formula_specialized(model, data)
    output = Vector{Float64}(undef, length(specialized))
    
    # Warmup
    for i in 1:10
        specialized(output, data, 1)
    end
    
    # Test total allocations
    total_allocs = @allocated begin
        for i in 1:100
            specialized(output, data, 1)
        end
    end
    
    println("Total allocations: $(total_allocs/100) bytes per call")
    
    # Break down by component type
    formula_data = specialized.data
    
    # Test functions alone
    if !isempty(formula_data.functions)
        func_allocs = @allocated begin
            for i in 1:100
                execute_linear_function_operations!(formula_data.functions, formula_data.function_scratch, output, data, 1)
            end
        end
        println("Functions alone: $(func_allocs/100) bytes per call")
    end
    
    # Test interactions alone
    if !isempty(formula_data.interactions)
        int_allocs = @allocated begin
            for i in 1:100
                execute_interaction_operations!(formula_data.interactions, formula_data.interaction_scratch, output, data, 1)
            end
        end
        println("Interactions alone: $(int_allocs/100) bytes per call")
    end
    
    # Calculate improvement
    original_specialized = 384.0  # From your profiling data
    improvement = (1 - (total_allocs/100) / original_specialized) * 100
    
    println("Improvement: $(round(improvement, digits=1))% reduction from previous specialized version")
    
    if total_allocs == 0
        println("🎉 ZERO ALLOCATIONS ACHIEVED!")
    elseif total_allocs/100 < 50
        println("🏆 EXCELLENT: Under 50 bytes per call!")
    elseif improvement > 50
        println("🥇 VERY GOOD: >50% improvement!")
    end
    
    return total_allocs / 100
end


test_comprehensive_fix(formula, df, data)