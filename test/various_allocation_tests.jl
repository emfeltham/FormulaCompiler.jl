function test_precomputed_operations(compiled, data)
    output = Vector{Float64}(undef, length(compiled))
    
    println("=== TESTING PRECOMPUTED OPERATIONS ===")
    
    # Test 1: Access precomputed constant operations
    const_ops = compiled.root_evaluator.constant_ops
    allocs1 = @allocated begin
        for op in const_ops
            typeof(op)
        end
    end
    println("Constant ops loop: $allocs1 bytes")
    
    # Test 2: Execute precomputed constant operations
    allocs2 = @allocated begin
        for op in const_ops
            output[op.position] = op.value
        end
    end
    println("Execute constant ops: $allocs2 bytes")
    
    # Test 3: Access precomputed continuous operations
    cont_ops = compiled.root_evaluator.continuous_ops
    allocs3 = @allocated begin
        for op in cont_ops
            typeof(op)
        end
    end
    println("Continuous ops loop: $allocs3 bytes")
    
    # Test 4: Execute precomputed continuous operations
    allocs4 = @allocated begin
        for op in cont_ops
            output[op.position] = Float64(data[op.column][1])
        end
    end
    println("Execute continuous ops: $allocs4 bytes")
    
    # Test 5: Manual single operation (should be 0)
    if !isempty(const_ops)
        allocs5 = @allocated begin
            op = const_ops[1]
            output[op.position] = op.value
        end
        println("Single constant op: $allocs5 bytes")
    end
    
    if !isempty(cont_ops)
        allocs6 = @allocated begin
            op = cont_ops[1]
            output[op.position] = Float64(data[op.column][1])
        end
        println("Single continuous op: $allocs6 bytes")
    end
end

test_precomputed_operations(compiled, data)

########

function analyze_evaluator_composition(compiled)
    println("=== EVALUATOR COMPOSITION ANALYSIS ===")
    
    root = compiled.root_evaluator
    println("Root evaluator type: $(typeof(root))")
    
    println("Constant ops: $(length(root.constant_ops))")
    println("Continuous ops: $(length(root.continuous_ops))")
    println("Categorical evaluators: $(length(root.categorical_evaluators))")
    println("Function evaluators: $(length(root.function_evaluators))")
    println("Interaction evaluators: $(length(root.interaction_evaluators))")
    
    # Show which ones might be causing allocations
    if !isempty(root.categorical_evaluators)
        println("Categorical evaluators present - potential allocation source")
    end
    if !isempty(root.function_evaluators)
        println("Function evaluators present - potential allocation source")
    end
    if !isempty(root.interaction_evaluators)
        println("Interaction evaluators present - potential allocation source")
    end
end

analyze_evaluator_composition(compiled)

#########

function diagnose_precomputed_types(compiled)
    println("=== PRECOMPUTED TYPE DIAGNOSIS ===")
    
    const_ops = compiled.root_evaluator.constant_ops
    cont_ops = compiled.root_evaluator.continuous_ops
    
    println("constant_ops type: $(typeof(const_ops))")
    println("continuous_ops type: $(typeof(cont_ops))")
    
    if !isempty(const_ops)
        println("First constant op type: $(typeof(const_ops[1]))")
        println("First constant op fields: position=$(const_ops[1].position), value=$(const_ops[1].value)")
    end
    
    if !isempty(cont_ops)
        println("First continuous op type: $(typeof(cont_ops[1]))")
        println("First continuous op fields: column=$(cont_ops[1].column), position=$(cont_ops[1].position)")
    end
end

diagnose_precomputed_types(compiled)

function test_vector_element_access(compiled, data)
    println("=== VECTOR ELEMENT ACCESS TEST ===")
    
    const_ops = compiled.root_evaluator.constant_ops
    cont_ops = compiled.root_evaluator.continuous_ops
    output = Vector{Float64}(undef, length(compiled))
    
    # Test 1: Just accessing vector elements
    if !isempty(const_ops)
        allocs1 = @allocated begin
            op = const_ops[1]
        end
        println("Access const_ops[1]: $allocs1 bytes")
        
        # Test 2: Accessing fields after getting the element
        op = const_ops[1]
        allocs2 = @allocated begin
            pos = op.position
            val = op.value
        end
        println("Access fields from local op: $allocs2 bytes")
        
        # Test 3: Direct inline access
        allocs3 = @allocated begin
            pos = const_ops[1].position
            val = const_ops[1].value
        end
        println("Direct field access const_ops[1].field: $allocs3 bytes")
    end
end

test_vector_element_access(compiled, data)

function test_completely_manual(compiled, data)
    println("=== COMPLETELY MANUAL TEST ===")
    
    output = Vector{Float64}(undef, length(compiled))
    
    # Manually hardcode the operations based on what we know
    allocs = @allocated begin
        # Manual constant: output[1] = 1.0 (assuming intercept)
        output[1] = 1.0
        
        # Manual continuous: output[2] = data.x[1], output[3] = data.y[1]
        output[2] = Float64(data.x[1])
        output[3] = Float64(data.y[1])
    end
    
    println("Completely manual hardcoded operations: $allocs bytes")
    
    return allocs
end

test_completely_manual(compiled, data)


################

function test_type_stability(compiled, data)
    println("=== TYPE STABILITY ANALYSIS ===")
    
    const_ops = compiled.root_evaluator.constant_ops
    cont_ops = compiled.root_evaluator.continuous_ops
    output = Vector{Float64}(undef, length(compiled))
    
    println("\n1. Testing vector access type stability:")
    println("@code_warntype for const_ops[1]:")
    @code_warntype const_ops[1]
    
    println("\n2. Testing field access type stability:")
    if !isempty(const_ops)
        op = const_ops[1]
        println("@code_warntype for op.position:")
        @code_warntype op.position
        
        println("@code_warntype for op.value:")
        @code_warntype op.value
    end
    
    println("\n3. Testing loop type stability:")
    function test_const_loop(ops, out)
        for op in ops
            out[op.position] = op.value
        end
    end
    
    println("@code_warntype for constant loop:")
    @code_warntype test_const_loop(const_ops, output)
    
    println("\n4. Testing continuous loop type stability:")
    function test_cont_loop(ops, out, data, row_idx)
        for op in ops
            out[op.position] = Float64(data[op.column][row_idx])
        end
    end
    
    println("@code_warntype for continuous loop:")
    @code_warntype test_cont_loop(cont_ops, output, data, 1)
    
    println("\n5. Testing single operation type stability:")
    function test_single_const(ops, out)
        if !isempty(ops)
            op = ops[1]
            out[op.position] = op.value
        end
    end
    
    println("@code_warntype for single constant operation:")
    @code_warntype test_single_const(const_ops, output)
end

test_type_stability(compiled, data)

#####

function test_global_vs_local(compiled, data)
    println("=== GLOBAL VS LOCAL VARIABLE TEST ===")
    
    output = Vector{Float64}(undef, length(compiled))
    
    # Test with global variables (current approach)
    const_ops = compiled.root_evaluator.constant_ops
    
    if !isempty(const_ops)
        allocs1 = @allocated begin
            op = const_ops[1]
            output[op.position] = op.value
        end
        println("Global variables: $allocs1 bytes")
        
        # Test with local variables
        function test_with_locals()
            local_compiled = compiled
            local_output = Vector{Float64}(undef, length(compiled))
            local_const_ops = local_compiled.root_evaluator.constant_ops
            
            @allocated begin
                op = local_const_ops[1]
                local_output[op.position] = op.value
            end
        end
        
        allocs2 = test_with_locals()
        println("Local variables: $allocs2 bytes")
    end
end

test_global_vs_local(compiled, data)

######## check if it is a data accessing issue

function test_type_stable_manual(compiled, data)
    output = Vector{Float64}(undef, length(compiled))
    
    allocs = @allocated begin
        # Manual type-stable operations
        output[1] = 1.0                    # Constant
        output[2] = Float64(data.x[1])     # Direct field access
        output[3] = Float64(data.y[1])     # Direct field access
    end
    
    println("Manual type-stable operations: $allocs bytes")
    return allocs
end

test_type_stable_manual(compiled, data)

### yes, it is

function test_manual_unrolled_execution(compiled, data)
    output = Vector{Float64}(undef, length(compiled))
    
    println("=== MANUAL UNROLLED EXECUTION TEST ===")
    
    # Get the segregated evaluators
    const_evals = compiled.root_evaluator.constant_evaluators
    cont_evals = compiled.root_evaluator.continuous_evaluators
    cat_evals = compiled.root_evaluator.categorical_evaluators
    func_evals = compiled.root_evaluator.function_evaluators
    interaction_evals = compiled.root_evaluator.interaction_evaluators
    
    # Test manual execution without any loops
    allocs = @allocated begin
        # Manual constant execution (no loop)
        if length(const_evals) >= 1
            output[const_evals[1].position] = const_evals[1].value
        end
        
        # Manual continuous execution (no loop)
        if length(cont_evals) >= 1
            eval = cont_evals[1]
            col = eval.column
            pos = eval.position
            output[pos] = Float64(data[col][1])
        end
        if length(cont_evals) >= 2
            eval = cont_evals[2]
            col = eval.column
            pos = eval.position
            output[pos] = Float64(data[col][1])
        end
        # Add more manual cases as needed...
    end
    
    println("Manual unrolled execution: $allocs bytes")
    
    return allocs
end

test_manual_unrolled_execution(compiled, data)

########

function isolate_allocation_source(compiled, data)
    output = Vector{Float64}(undef, length(compiled))
    
    println("=== ISOLATING ALLOCATION SOURCES ===")
    
    # Test 1: Just array access
    allocs1 = @allocated begin
        output[1] = 1.0
        output[2] = 2.0
    end
    println("Basic array assignment: $allocs1 bytes")
    
    # Test 2: Data access only
    allocs2 = @allocated begin
        val = data.x[1]
        val2 = data.y[1]
    end
    println("Data access only: $allocs2 bytes")
    
    # Test 3: Combined data access + assignment
    allocs3 = @allocated begin
        output[1] = Float64(data.x[1])
        output[2] = Float64(data.y[1])
    end
    println("Data access + assignment: $allocs3 bytes")
    
    # Test 4: Field access from evaluators
    if !isempty(compiled.root_evaluator.continuous_evaluators)
        eval = compiled.root_evaluator.continuous_evaluators[1]
        allocs4 = @allocated begin
            col = eval.column
            pos = eval.position
        end
        println("Evaluator field access: $allocs4 bytes")
        
        # Test 5: Complete operation breakdown
        allocs5 = @allocated begin
            col = eval.column
            pos = eval.position
            val = data[col][1]
            output[pos] = Float64(val)
        end
        println("Complete operation: $allocs5 bytes")
    end
end

isolate_allocation_source(compiled, data)

####

function test_with_local_variables(compiled, data)
    println("=== TESTING WITH LOCAL VARIABLES ===")
    
    # Make everything local to avoid global variable issues
    local_compiled = compiled
    local_data = data
    local_output = Vector{Float64}(undef, length(compiled))
    
    allocs = @allocated begin
        # Single operation with all local variables
        local_output[1] = Float64(local_data.x[1])
    end
    
    println("Single operation (all local): $allocs bytes")
    
    # Test the compiled formula call with local variables
    allocs2 = @allocated local_compiled(local_output, local_data, 1)
    println("CompiledFormula call (all local): $allocs2 bytes")
end

test_with_local_variables(compiled, data)

########

function test_continuous_ops_after_fix(compiled, data)
    println("=== TESTING CONTINUOUS OPS AFTER TYPE-STABLE FIX ===")
    
    output = Vector{Float64}(undef, length(compiled))
    cont_ops = compiled.root_evaluator.continuous_ops
    
    # Test the fixed continuous operations
    allocs = @allocated begin
        for op in cont_ops
            col = op.column
            pos = op.position
            
            val = if col === :x
                data.x[1]
            elseif col === :y
                data.y[1] 
            elseif col === :z
                data.z[1]
            else
                data[col][1]  # Fallback
            end
            
            output[pos] = Float64(val)
        end
    end
    
    println("Fixed continuous ops: $allocs bytes")
    
    # Compare with the old approach
    allocs_old = @allocated begin
        for op in cont_ops
            output[op.position] = Float64(data[op.column][1])
        end
    end
    
    println("Old continuous ops: $allocs_old bytes")
end

test_continuous_ops_after_fix(compiled, data)

using FormulaCompiler: execute_function_self_contained!,execute_interaction_self_contained!
function identify_remaining_allocations(compiled, data)
    println("=== IDENTIFYING REMAINING ALLOCATION SOURCES ===")
    
    output = Vector{Float64}(undef, length(compiled))
    scratch = compiled.scratch_space
    
    # Test each evaluator type separately
    println("Testing individual evaluator types:")
    
    # 1. Constants (should be 0 now)
    const_ops = compiled.root_evaluator.constant_ops
    allocs1 = @allocated begin
        for op in const_ops
            output[op.position] = op.value
        end
    end
    println("  Constants: $allocs1 bytes")
    
    # 2. Continuous (should be much better now)
    cont_ops = compiled.root_evaluator.continuous_ops
    allocs2 = @allocated begin
        for op in cont_ops
            col = op.column
            pos = op.position
            val = if col === :x
                data.x[1]
            elseif col === :y
                data.y[1]
            else
                data[col][1]
            end
            output[pos] = Float64(val)
        end
    end
    println("  Continuous (fixed): $allocs2 bytes")
    
    # 3. Categorical (likely still allocating)
    cat_evals = compiled.root_evaluator.categorical_evaluators
    allocs3 = @allocated begin
        for eval in cat_evals
            level_codes = eval.level_codes
            cm = eval.contrast_matrix
            positions = eval.positions
            n_levels = eval.n_levels
            
            lvl = level_codes[1]
            lvl = lvl < 1 ? 1 : (lvl > n_levels ? n_levels : lvl)
            
            for j in 1:length(positions)
                output[positions[j]] = cm[lvl, j]
            end
        end
    end
    println("  Categorical: $allocs3 bytes")
    
    # 4. Functions (likely still allocating)
    func_evals = compiled.root_evaluator.function_evaluators
    allocs4 = @allocated begin
        for eval in func_evals
            execute_function_self_contained!(eval, scratch, output, data, 1)
        end
    end
    println("  Functions: $allocs4 bytes")
    
    # 5. Interactions (likely the biggest source)
    interaction_evals = compiled.root_evaluator.interaction_evaluators
    allocs5 = @allocated begin
        for eval in interaction_evals
            execute_interaction_self_contained!(eval, scratch, output, data, 1)
        end
    end
    println("  Interactions: $allocs5 bytes")
    
    println("\nTotal individual: $(allocs1 + allocs2 + allocs3 + allocs4 + allocs5) bytes")
end

identify_remaining_allocations(compiled, data)

######### interactions

function debug_interaction_allocations(compiled, data)
    println("=== DEBUGGING INTERACTION ALLOCATIONS ===")
    
    output = Vector{Float64}(undef, length(compiled))
    scratch = compiled.scratch_space
    interaction_evals = compiled.root_evaluator.interaction_evaluators
    
    for (i, eval) in enumerate(interaction_evals)
        println("\nInteraction $i:")
        println("  Type: $(typeof(eval))")
        println("  Components: $(length(eval.components))")
        
        # Test just this interaction
        allocs = @allocated execute_interaction_self_contained!(eval, scratch, output, data, 1)
        println("  Allocations: $allocs bytes")
        
        # Test the components individually
        println("  Component breakdown:")
        for (j, comp) in enumerate(eval.components)
            comp_allocs = @allocated begin
                comp_range = eval.component_scratch_map[j]
                comp_start = first(comp_range)
                
                if comp isa FormulaCompiler.ContinuousEvaluator
                    scratch[comp_start] = Float64(data[comp.column][1])
                elseif comp isa FormulaCompiler.ConstantEvaluator
                    scratch[comp_start] = comp.value
                elseif comp isa FormulaCompiler.CategoricalEvaluator
                    # Manual categorical execution
                    level_codes = comp.level_codes
                    cm = comp.contrast_matrix
                    n_levels = comp.n_levels
                    
                    lvl = level_codes[1]
                    lvl = lvl < 1 ? 1 : (lvl > n_levels ? n_levels : lvl)
                    
                    comp_end = last(comp_range)
                    n_contrasts = comp_end - comp_start + 1
                    for k in 1:n_contrasts
                        scratch[comp_start + k - 1] = cm[lvl, k]
                    end
                end
            end
            println("    Component $j ($(typeof(comp))): $comp_allocs bytes")
        end
        
        # Test just the Kronecker application
        kronecker_allocs = @allocated FormulaCompiler.apply_kronecker_pattern_to_positions!(
            eval.kronecker_pattern,
            eval.component_scratch_map,
            scratch,
            output,
            eval.positions
        )
        println("  Kronecker application: $kronecker_allocs bytes")
    end
end

debug_interaction_allocations(compiled, data)

#####

function debug_categorical_in_interactions(compiled, data)
    println("=== DEBUGGING CATEGORICAL IN INTERACTIONS ===")
    
    scratch = compiled.scratch_space
    interaction_evals = compiled.root_evaluator.interaction_evaluators
    
    for (i, eval) in enumerate(interaction_evals)
        for (j, comp) in enumerate(eval.components)
            if comp isa FormulaCompiler.CategoricalEvaluator
                println("\nInteraction $i, Component $j (Categorical):")
                println("  Column: $(comp.column)")
                println("  Level codes length: $(length(comp.level_codes))")
                println("  Level codes sample: $(comp.level_codes[1:min(5, end)])")
                println("  N levels: $(comp.n_levels)")
                println("  Positions: $(comp.positions)")
                
                # Test just the level code access
                level_codes = comp.level_codes
                allocs1 = @allocated begin
                    lvl = level_codes[1]
                end
                println("  Level code access: $allocs1 bytes")
                
                # Test the contrast matrix access
                cm = comp.contrast_matrix
                allocs2 = @allocated begin
                    val = cm[1, 1]
                end
                println("  Contrast matrix access: $allocs2 bytes")
                
                # Test the full categorical operation manually
                comp_range = eval.component_scratch_map[j]
                comp_start = first(comp_range)
                comp_end = last(comp_range)
                
                allocs3 = @allocated begin
                    lvl = level_codes[1]
                    lvl = lvl < 1 ? 1 : (lvl > comp.n_levels ? comp.n_levels : lvl)
                    
                    n_contrasts = comp_end - comp_start + 1
                    for k in 1:n_contrasts
                        scratch[comp_start + k - 1] = cm[lvl, k]
                    end
                end
                println("  Full categorical operation: $allocs3 bytes")
                
                # Test if it's the loop vs direct access
                allocs4 = @allocated begin
                    lvl = level_codes[1]
                    scratch[comp_start] = cm[lvl, 1]
                    if comp_end > comp_start
                        scratch[comp_start + 1] = cm[lvl, 2]
                    end
                end
                println("  Direct access (no loop): $allocs4 bytes")
            end
        end
    end
end

debug_categorical_in_interactions(compiled, data)


#######

# Test 1: Is it the CompiledFormula operator itself?
function test_call_operator_isolation(compiled, data)
    output = Vector{Float64}(undef, length(compiled))
    root_eval = compiled.root_evaluator
    scratch = compiled.scratch_space
    
    # Direct evaluator call (bypass CompiledFormula operator)
    allocs1 = @allocated FormulaCompiler.execute_self_contained!(root_eval, scratch, output, data, 1)
    println("Direct evaluator call: $allocs1 bytes")
    
    # CompiledFormula operator call
    allocs2 = @allocated compiled(output, data, 1)
    println("CompiledFormula operator: $allocs2 bytes")
end

# Test 2: Is it method compilation?
function test_compilation_overhead(compiled, data)
    output = Vector{Float64}(undef, length(compiled))
    
    # Many warmup calls
    for _ in 1:10000
        compiled(output, data, 1)
    end
    
    # Test after extensive warmup
    allocs = @allocated compiled(output, data, 1)
    println("After 10k warmup calls: $allocs bytes")
end

# Test 3: Is it the benchmark itself?
function test_benchmark_overhead()
    x = 1.0
    y = 2.0
    z = Vector{Float64}(undef, 5)
    
    # Simple operations that should be 0 bytes
    allocs = @allocated begin
        z[1] = x + y
        z[2] = x * y
    end
    println("Simple operations: $allocs bytes")
end

test_call_operator_isolation(compiled, data)
test_compilation_overhead(compiled, data)
test_benchmark_overhead()

#######

function isolate_evaluator_allocation_sources(compiled, data)
    println("=== ISOLATING EXACT ALLOCATION SOURCES ===")
    
    output = Vector{Float64}(undef, length(compiled))
    scratch = compiled.scratch_space
    root = compiled.root_evaluator
    
    # Test each evaluator group in isolation with everything local
    function test_constants_only()
        local_output = Vector{Float64}(undef, length(compiled))
        local_ops = root.constant_ops
        @allocated begin
            @inbounds for op in local_ops
                local_output[op.position] = op.value
            end
        end
    end
    
    function test_continuous_only()
        local_output = Vector{Float64}(undef, length(compiled))
        local_ops = root.continuous_ops
        local_data = data
        @allocated begin
            @inbounds for op in local_ops
                col = op.column
                pos = op.position
                val = if col === :x
                    local_data.x[1]
                elseif col === :y
                    local_data.y[1]
                else
                    local_data[col][1]
                end
                local_output[pos] = Float64(val)
            end
        end
    end
    
    function test_empty_execution()
        local_output = Vector{Float64}(undef, length(compiled))
        @allocated begin
            # Do literally nothing
            nothing
        end
    end
    
    println("Empty execution: $(test_empty_execution()) bytes")
    println("Constants only: $(test_constants_only()) bytes") 
    println("Continuous only: $(test_continuous_only()) bytes")
    
    # Test if it's the @inbounds loops themselves
    function test_loop_structures()
        local_ops = root.constant_ops
        @allocated begin
            @inbounds for op in local_ops
                # Do nothing with op
                typeof(op)
            end
        end
    end
    
    println("Loop structure only: $(test_loop_structures()) bytes")
end

isolate_evaluator_allocation_sources(compiled, data)

######

function test_without_complex_evaluators(compiled, data)
    println("=== TESTING WITHOUT COMPLEX EVALUATORS ===")
    
    output = Vector{Float64}(undef, length(compiled))
    scratch = compiled.scratch_space
    root = compiled.root_evaluator
    
    # Test execution with ONLY the zero-allocation parts
    allocs = @allocated begin
        # Only constants and continuous (we proved these are 0-allocation)
        @inbounds for op in root.constant_ops
            output[op.position] = op.value
        end
        
        @inbounds for op in root.continuous_ops
            col = op.column
            pos = op.position
            val = if col === :x
                data.x[1]
            elseif col === :y
                data.y[1]
            else
                data[col][1]
            end
            output[pos] = Float64(val)
        end
        
        # SKIP: categorical_evaluators, function_evaluators, interaction_evaluators
    end
    
    println("Only simple operations (skip complex): $allocs bytes")
    
    # This should be 0 bytes, confirming our hypothesis
    return allocs
end

test_without_complex_evaluators(compiled, data)

########

function test_execution_infrastructure(compiled, data)
    println("=== TESTING EXECUTION INFRASTRUCTURE ===")
    
    output = Vector{Float64}(undef, length(compiled))
    scratch = compiled.scratch_space
    root = compiled.root_evaluator
    
    # Test 1: Direct operations without ANY method calls
    allocs1 = @allocated begin
        # Completely manual, no method dispatch
        output[1] = 1.0  # Manual constant
        output[2] = Float64(data.x[1])  # Manual continuous
        output[3] = Float64(data.y[1])  # Manual continuous
        # Don't touch any other positions
    end
    println("Completely manual operations: $allocs1 bytes")
    
    # Test 2: Call the method but with empty body
    function empty_execute_self_contained!(evaluator, scratch, output, data, row_idx)
        # Do absolutely nothing
        return nothing
    end
    
    allocs2 = @allocated empty_execute_self_contained!(root, scratch, output, data, 1)
    println("Empty method call: $allocs2 bytes")
    
    # Test 3: Just accessing the evaluator fields
    allocs3 = @allocated begin
        const_ops = root.constant_ops
        cont_ops = root.continuous_ops
        cat_evals = root.categorical_evaluators
        func_evals = root.function_evaluators
        interaction_evals = root.interaction_evaluators
    end
    println("Just accessing evaluator fields: $allocs3 bytes")
    
    # Test 4: The @inbounds for loops with no body
    allocs4 = @allocated begin
        @inbounds for op in root.constant_ops
            # Do nothing
        end
        @inbounds for op in root.continuous_ops  
            # Do nothing
        end
        @inbounds for eval in root.categorical_evaluators
            # Do nothing  
        end
        @inbounds for eval in root.function_evaluators
            # Do nothing
        end
        @inbounds for eval in root.interaction_evaluators
            # Do nothing
        end
    end
    println("Empty loops over all evaluator collections: $allocs4 bytes")
end

test_execution_infrastructure(compiled, data)

###

# Test each component separately:
compiled = compile_formula(model, data)
output = Vector{Float64}(undef, length(compiled))

# 1. Test just constants + continuous (should be 0)
println("Constants + Continuous only:")
@allocated begin
    for op in compiled.root_evaluator.constant_ops
        output[op.position] = op.value
    end
    for op in compiled.root_evaluator.continuous_ops
        col = op.column
        pos = op.position
        val = if col === :x
            data.x[1]
        elseif col === :y
            data.y[1]
        else
            data[col][1]
        end
        output[pos] = Float64(val)
    end
end

# 2. Test just categorical loop (should be small)
println("Categorical loop only:")
@allocated begin
    for eval in compiled.root_evaluator.categorical_evaluators
        # Empty loop to test iteration overhead
    end
end

# 3. Test just interaction loop 
println("Interaction loop only:")
@allocated begin
    for eval in compiled.root_evaluator.interaction_evaluators
        # Empty loop to test iteration overhead  
    end
end

####

# Test 1: Just constants (should be absolute zero)
println("Constants only:")
@allocated begin
    for op in compiled.root_evaluator.constant_ops
        output[op.position] = op.value
    end
end

# Test 2: Just continuous (test our type-stable fix)
println("Continuous only:")
@allocated begin
    for op in compiled.root_evaluator.continuous_ops
        col = op.column
        pos = op.position
        val = if col === :x
            data.x[1]
        elseif col === :y
            data.y[1]
        else
            data[col][1]
        end
        output[pos] = Float64(val)
    end
end

# Test 3: Manual constant operation (bypass loop)
println("Single manual constant:")
if !isempty(compiled.root_evaluator.constant_ops)
    op = compiled.root_evaluator.constant_ops[1]
    @allocated begin
        output[op.position] = op.value
    end
end

# Test 4: Manual continuous operation (bypass loop)
println("Single manual continuous:")
if !isempty(compiled.root_evaluator.continuous_ops)
    op = compiled.root_evaluator.continuous_ops[1]
    @allocated begin
        output[op.position] = Float64(data.x[1])  # Hardcode known column
    end
end

###

# Test 5: Pure array assignment (should be absolute zero)
println("Pure array assignment:")
@allocated begin
    output[1] = 1.0
    output[2] = 2.0
end

# Test 6: Pure data access (should be absolute zero)  
println("Pure data access:")
@allocated begin
    val1 = data.x[1]
    val2 = data.y[1]
end

# Test 7: Field access from precomputed op (the smoking gun?)
if !isempty(compiled.root_evaluator.constant_ops)
    op = compiled.root_evaluator.constant_ops[1]
    println("Field access from constant op:")
    @allocated begin
        pos = op.position
        val = op.value
    end
    
    println("Combined field access + assignment:")
    @allocated begin
        pos = op.position
        val = op.value
        output[pos] = val
    end
end

# Test 8: Test if it's the evaluator field access
println("Direct evaluator field access:")
@allocated begin
    ops = compiled.root_evaluator.constant_ops
    continuous_ops = compiled.root_evaluator.continuous_ops
end

###

function test_with_all_locals()
    local_compiled = compiled
    local_data = data
    local_output = Vector{Float64}(undef, length(compiled))
    
    @allocated local_compiled(local_output, local_data, 1)
end

test_with_all_locals()

function test_preextracted_fields()
    # Extract everything at function entry
    const_ops = compiled.root_evaluator.constant_ops
    cont_ops = compiled.root_evaluator.continuous_ops
    
    @allocated begin
        # Now use local variables instead of field access
        for op in const_ops
            output[op.position] = op.value
        end
    end
end

test_preextracted_fields()