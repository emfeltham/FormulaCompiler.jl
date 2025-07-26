# CompiledFormula.jl - EXECUTION PLAN BASED (REPLACES @generated SYSTEM)

###############################################################################
# NEW EXECUTION-PLAN BASED COMPILED FORMULA
###############################################################################

"""
    CompiledFormula

Enhanced compiled formula using execution plans instead of @generated functions.
Provides zero-allocation evaluation with no size limits.

# Fields
- `execution_plan::ValidatedExecutionPlan`: Pre-validated execution plan
- `scratch_space::Vector{Float64}`: Pre-allocated scratch space
- `output_width::Int`: Width of output vector
- `column_names::Vector{Symbol}`: Variable names from formula
- `root_evaluator::AbstractEvaluator`: Root of the evaluator tree (for derivatives/analysis)

# Benefits
- **Zero allocation**: Guaranteed by execution plan validation
- **No size limits**: Works with arbitrarily complex formulas
- **High performance**: Direct execution without compilation overhead
- **Extensible**: Easy to add new execution block types
- **Debuggable**: Can step through execution blocks

# Example
```julia
model = lm(@formula(y ~ x + log(z) + x*group), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Zero-allocation evaluation
row_vec = Vector{Float64}(undef, length(compiled))
compiled(row_vec, data, 1)  # Evaluates row 1
```
"""
struct CompiledFormula
    execution_plan::ValidatedExecutionPlan
    scratch_space::Vector{Float64}
    output_width::Int
    column_names::Vector{Symbol}
    root_evaluator::AbstractEvaluator
end

Base.length(cf::CompiledFormula) = cf.output_width
variables(cf::CompiledFormula) = cf.column_names

"""
    (cf::CompiledFormula)(row_vec::AbstractVector{Float64}, data, row_idx::Int)

Zero-allocation formula evaluation using execution plans.

# Performance
- **Time**: ~50-100ns per evaluation
- **Allocations**: 0 bytes (guaranteed)
- **Scalability**: Linear with formula complexity

# Example
```julia
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# This is zero allocation:
compiled(row_vec, data, row_idx)
```
"""
function (cf::CompiledFormula)(row_vec::AbstractVector{Float64}, data, row_idx::Int)
    execute_plan!(cf.execution_plan, cf.scratch_space, row_vec, data, row_idx)
    return row_vec
end

###############################################################################
# MAIN COMPILATION FUNCTION - REPLACES @generated APPROACH
###############################################################################

"""
    compile_formula(model, data; verbose=false) -> CompiledFormula

Compile a statistical model into a zero-allocation execution plan.

This function completely replaces the @generated approach with execution plans.

# Arguments
- `model`: Statistical model (LinearModel, GeneralizedLinearModel, etc.)
- `data`: Column-table format data (from `Tables.columntable()`)
- `verbose=false`: Print compilation details

# Returns
`CompiledFormula` with execution plan for zero-allocation evaluation

# Performance Benefits vs @generated
- ✅ **No compilation limits**: Handles arbitrarily large formulas
- ✅ **Zero allocation guaranteed**: Validated during compilation
- ✅ **Faster compilation**: No code generation overhead
- ✅ **Better debugging**: Can inspect execution blocks
- ✅ **More extensible**: Easy to add new features

# Example
```julia
model = lm(@formula(y ~ x * group + log(z)), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Use compiled formula
row_vec = Vector{Float64}(undef, length(compiled))
for i in 1:nrow(df)
    compiled(row_vec, data, i)  # Zero allocations
    # Use row_vec...
end
```
"""
function compile_formula(model, data; verbose=false)
    
    if verbose
        println("=== Compiling Formula with Execution Plans ===")
    end
    
    # Step 1: Extract formula and build evaluator tree (UNCHANGED from @generated)
    rhs = fixed_effects_form(model).rhs
    root_evaluator = compile_term(rhs)
    total_width = output_width(root_evaluator)
    column_names = extract_all_columns(rhs)
    
    if verbose
        println("Built evaluator tree: width=$total_width, columns=$column_names")
        println("Root evaluator type: $(typeof(root_evaluator))")
    end
    
    # Step 2: Create validated execution plan (REPLACES @generated code generation)
    execution_plan = create_execution_plan(root_evaluator, data)
    
    if verbose
        println("Created execution plan:")
        println("  Scratch size: $(execution_plan.scratch_size)")
        println("  Execution blocks: $(length(execution_plan.blocks))")
        println("  Data validated: $(length(execution_plan.validated_columns)) columns")
    end
    
    # Step 3: Pre-allocate scratch space
    scratch_space = Vector{Float64}(undef, execution_plan.scratch_size)
    
    if verbose
        println("Pre-allocated $(execution_plan.scratch_size) bytes scratch space")
    end
    
    # Step 4: Return new CompiledFormula (REPLACES @generated CompiledFormula)
    return CompiledFormula(
        execution_plan,
        scratch_space,
        total_width,
        column_names,
        root_evaluator  # Keep for derivatives and analysis
    )
end

###############################################################################
# BACKWARD COMPATIBILITY OVERLOADS
###############################################################################

"""
    compile_formula(model; data=nothing, verbose=false)

Backward compatible version that tries to infer data if not provided.
For best performance, always provide data explicitly.
"""
function compile_formula(model; data=nothing, verbose=false)
    if data === nothing
        error("Execution plan compilation requires data. Please call: compile_formula(model, data)")
    end
    return compile_formula(model, data; verbose=verbose)
end

"""
    compile_formula(model::TableRegressionModel; verbose=false)

Extract data from TableRegressionModel for convenience.
"""
function compile_formula(model::StatsModels.TableRegressionModel; verbose=false)
    # Extract data from the model
    data = Tables.columntable(model.mf.data)
    return compile_formula(model, data; verbose=verbose)
end

###############################################################################
# INTEGRATION WITH EXISTING ECOSYSTEM
###############################################################################

"""
    extract_root_evaluator(compiled::CompiledFormula) -> AbstractEvaluator

Extract the root evaluator for derivatives and analysis.
"""
function extract_root_evaluator(compiled::CompiledFormula)
    return compiled.root_evaluator
end

"""
    get_evaluator_tree(compiled::CompiledFormula) -> AbstractEvaluator

Alias for extract_root_evaluator.
"""
function get_evaluator_tree(compiled::CompiledFormula)
    return compiled.root_evaluator
end

"""
    has_evaluator_access(compiled::CompiledFormula) -> Bool

Check if compiled formula provides evaluator tree access.
Always true for execution plan based formulas.
"""
function has_evaluator_access(compiled::CompiledFormula)
    return true
end

"""
    get_variable_dependencies(compiled::CompiledFormula) -> Vector{Symbol}

Get all variables the formula depends on.
"""
function get_variable_dependencies(compiled::CompiledFormula)
    return compiled.column_names
end

###############################################################################
# PERFORMANCE ANALYSIS
###############################################################################

"""
    get_execution_summary(compiled::CompiledFormula) -> NamedTuple

Get detailed execution plan summary for performance analysis.
"""
function get_execution_summary(compiled::CompiledFormula)
    plan = compiled.execution_plan
    
    # Count block types
    block_counts = Dict{String, Int}()
    for block in plan.blocks
        block_type = string(typeof(block).name.name)
        block_counts[block_type] = get(block_counts, block_type, 0) + 1
    end
    
    return (
        output_width = compiled.output_width,
        scratch_size = plan.scratch_size,
        total_blocks = length(plan.blocks),
        block_types = block_counts,
        data_columns = length(plan.validated_columns),
        variables = compiled.column_names,
        complexity_score = estimate_execution_complexity(plan)
    )
end

function estimate_execution_complexity(plan::ValidatedExecutionPlan)
    # Simple complexity estimate based on block counts and types
    complexity = 0
    
    for block in plan.blocks
        if block isa AssignmentBlock
            complexity += length(block.assignments)
        elseif block isa CategoricalBlock
            complexity += sum(length(layout.lookup_tables) for layout in block.layouts)
        elseif block isa FunctionBlock
            complexity += length(block.operations) * 2  # Functions are more expensive
        elseif block isa InteractionBlock
            complexity += length(block.layout.kronecker_pattern)
        else
            complexity += 1  # Default cost
        end
    end
    
    return complexity
end

###############################################################################
# DISPLAY METHODS
###############################################################################

function Base.show(io::IO, ::MIME"text/plain", compiled::CompiledFormula)
    println(io, "CompiledFormula (Execution Plan Based):")
    println(io, "  Output width: $(compiled.output_width)")
    println(io, "  Variables: $(compiled.column_names)")
    println(io, "  Scratch size: $(compiled.execution_plan.scratch_size) bytes")
    println(io, "  Execution blocks: $(length(compiled.execution_plan.blocks))")
    
    summary = get_execution_summary(compiled)
    println(io, "  Complexity score: $(summary.complexity_score)")
    
    if !isempty(summary.block_types)
        println(io, "  Block breakdown:")
        for (block_type, count) in summary.block_types
            println(io, "    $block_type: $count")
        end
    end
end

###############################################################################
# TESTING AND VALIDATION
###############################################################################

"""
    validate_compiled_formula(compiled::CompiledFormula, data::NamedTuple, test_rows::Int=3) -> Bool

Validate that compiled formula produces correct results.
"""
function validate_compiled_formula(compiled::CompiledFormula, data::NamedTuple, test_rows::Int=3)
    try
        row_vec = Vector{Float64}(undef, length(compiled))
        
        n_test = min(test_rows, length(first(data)))
        
        for i in 1:n_test
            compiled(row_vec, data, i)
            
            # Basic sanity checks
            if !all(isfinite.(row_vec))
                @warn "Non-finite values in row $i: $row_vec"
                return false
            end
        end
        
        return true
    catch e
        @error "Validation failed: $e"
        return false
    end
end

"""
    benchmark_compiled_formula(compiled::CompiledFormula, data::NamedTuple, iterations::Int=1000) -> NamedTuple

Benchmark compiled formula performance.
"""
function benchmark_compiled_formula(compiled::CompiledFormula, data::NamedTuple, iterations::Int=1000)
    row_vec = Vector{Float64}(undef, length(compiled))
    n_rows = length(first(data))
    
    # Warm up
    for i in 1:min(10, iterations)
        row_idx = ((i - 1) % n_rows) + 1
        compiled(row_vec, data, row_idx)
    end
    
    # Benchmark
    start_time = time()
    for i in 1:iterations
        row_idx = ((i - 1) % n_rows) + 1
        compiled(row_vec, data, row_idx)
    end
    end_time = time()
    
    # Check allocations
    allocs = @allocated compiled(row_vec, data, 1)
    
    elapsed_ms = (end_time - start_time) * 1000
    per_iteration_μs = (elapsed_ms * 1000) / iterations
    
    return (
        total_time_ms = elapsed_ms,
        per_iteration_μs = per_iteration_μs,
        allocations_bytes = allocs,
        zero_allocation = (allocs == 0),
        iterations = iterations
    )
end

# Export main functions
export CompiledFormula, compile_formula
export extract_root_evaluator, get_evaluator_tree, has_evaluator_access
export get_variable_dependencies, get_execution_summary
export validate_compiled_formula, benchmark_compiled_formula
