# CompiledFormula.jl - CLEAN VERSION - No backward compatibility
# Pure execution plan system, simplified API


###############################################################################
# CLEAN COMPILEDFORMULA STRUCT
###############################################################################

"""
    CompiledFormula

A compiled formula. Requires explicit data for compilation and validation.

# Fields
- `root_evaluator::AbstractEvaluator`: Evaluator tree
- `scratch_space::Vector{Float64}`: Pre-allocated scratch space
- `output_width::Int`: Number of output columns
- `column_names::Vector{Symbol}`: Names of all columns referenced
- `categorical_levels::Dict{Symbol, Vector{Int}}`: Pre-extracted levels

# Example
```julia
model = lm(@formula(y ~ x * group), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

output = Vector{Float64}(undef, length(compiled))
for i in 1:nrow(df)
    compiled(output, data, i)
end
```
"""
struct CompiledFormula
    root_evaluator::AbstractEvaluator    # Self-contained evaluator tree
    scratch_space::Vector{Float64}       # Pre-allocated scratch space
    output_width::Int                    # Number of output columns
    column_names::Vector{Symbol}         # Names of referenced columns
    categorical_levels::Dict{Symbol, Vector{Int}}  # Pre-extracted levels
end

###############################################################################
# COMPILATION FUNCTION
###############################################################################

"""
    compile_formula(model, data::NamedTuple) -> CompiledFormula

Compile formula using schema-based categorical extraction.
UPDATED: Now uses fitted model's schema for categorical contrasts.
"""
function compile_formula(model, data::NamedTuple)
    rhs = fixed_effects_form(model).rhs
    column_names = extract_all_columns(rhs)
    
    # println("DEBUG: compile_formula called")
    # println("DEBUG: Columns referenced: $column_names")
    
    # Use the updated schema extraction
    categorical_schema = extract_complete_categorical_schema(model)
    
    # Populate schema with actual data level codes
    populate_level_codes_from_data!(categorical_schema, data)
    
    # Determine which variables have main effects vs interaction-only
    determine_main_effect_contrasts!(categorical_schema, model)
    
    # Validate the extracted schema
    validate_categorical_schema(categorical_schema)
    
    # println("DEBUG: Schema extraction complete, $(length(categorical_schema)) categorical variables")
    
    # Continue with normal compilation...
    root_evaluator = compile_term(rhs, 1, ScratchAllocator(), categorical_schema)

# println("DEBUG: root_evaluator type: $(typeof(root_evaluator))")
# println("DEBUG: root_evaluator is nothing: $(root_evaluator === nothing)")

if root_evaluator === nothing
    error("compile_term returned nothing - check compilation logic")
end

out_width = output_width(root_evaluator)
    
    scratch_size = max_scratch_needed(root_evaluator)
    
    validate_data_columns!(data, column_names)
    
    # Extract level codes for backward compatibility
    categorical_levels = Dict{Symbol, Vector{Int}}()
    for (col_name, schema_info) in categorical_schema
        categorical_levels[col_name] = schema_info.level_codes
    end
    
    # println("DEBUG: Compilation complete - output_width: $out_width, scratch_size: $scratch_size")
    
    return CompiledFormula(
        root_evaluator,
        Vector{Float64}(undef, scratch_size),
        out_width,
        column_names,
        categorical_levels
    )
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

"""
    validate_data_columns!(data::NamedTuple, required_columns::Vector{Symbol})

Validate that data contains all columns referenced by the evaluator.
"""
function validate_data_columns!(data::NamedTuple, required_columns::Vector{Symbol})
    data_columns = keys(data)
    missing_columns = setdiff(required_columns, data_columns)
    
    if !isempty(missing_columns)
        throw(ArgumentError("Missing required columns in data: $(missing_columns). Available: $(collect(data_columns))"))
    end
    
    # Validate data consistency (all columns same length)
    if !isempty(data_columns)
        data_length = length(first(data))
        for col in data_columns
            if length(data[col]) != data_length
                throw(ArgumentError("Inconsistent data: column $col has length $(length(data[col])), expected $data_length"))
            end
        end
    end
    
    return nothing
end

"""
    Base.length(cf::CompiledFormula) -> Int

Get the output width of a compiled formula.
"""
Base.length(cf::CompiledFormula) = cf.output_width

"""
    get_scratch_size(cf::CompiledFormula) -> Int

Get the scratch space size required by this compiled formula.
"""
get_scratch_size(cf::CompiledFormula) = length(cf.scratch_space)

"""
    get_column_names(cf::CompiledFormula) -> Vector{Symbol}

Get the column names referenced by this compiled formula.
"""
get_column_names(cf::CompiledFormula) = cf.column_names

"""
    get_evaluator_tree(cf::CompiledFormula) -> AbstractEvaluator

Get the root evaluator tree.
"""
get_evaluator_tree(cf::CompiledFormula) = cf.root_evaluator

# Updated benchmark function:
"""
    benchmark_execution(cf::CompiledFormula, data::NamedTuple, n_iterations::Int = 1000) -> NamedTuple

Benchmark self-contained execution performance.
"""
function benchmark_execution(cf::CompiledFormula, data::NamedTuple, n_iterations::Int = 1000)
    output = Vector{Float64}(undef, cf.output_width)
    data_length = length(first(data))
    
    # Warmup
    for _ in 1:10
        cf(output, data, 1)
    end
    
    # Benchmark timing
    elapsed = @elapsed begin
        for i in 1:n_iterations
            row_idx = ((i - 1) % data_length) + 1
            cf(output, data, row_idx)
        end
    end
    
    # Check allocations
    alloc = @allocated begin
        for i in 1:100
            row_idx = ((i - 1) % data_length) + 1
            cf(output, data, row_idx)
        end
    end
    
    avg_time_ns = (elapsed / n_iterations) * 1e9
    avg_alloc = alloc / 100
    is_zero = avg_alloc == 0
    
    println("Self-Contained Evaluator Performance:")
    println("  Average time: $(round(avg_time_ns, digits=1)) ns")
    println("  Average allocations: $(avg_alloc) bytes")
    println("  Zero allocation: $(is_zero ? "✅ YES" : "❌ NO")")
    
    return (time_ns = avg_time_ns, allocations = avg_alloc, zero_allocation = is_zero)
end


"""
    prepare_categorical_levels(data::NamedTuple, column_names::Vector{Symbol}) -> Dict{Symbol, Vector{Int}}
    
DEPRECATED: Use extract_complete_categorical_schema instead.
Kept for backward compatibility during transition.
"""
function prepare_categorical_levels(data::NamedTuple, column_names::Vector{Symbol})
    @warn "prepare_categorical_levels is deprecated - schema extraction should handle this"
    
    level_maps = Dict{Symbol, Vector{Int}}()
    
    for col_name in column_names
        if haskey(data, col_name)
            col_data = data[col_name]
            if col_data isa CategoricalVector
                # Pre-extract all level codes - allocate once during compilation
                level_maps[col_name] = [levelcode(val) for val in col_data]
            end
        end
    end
    
    return level_maps
end

"""
    extract_level_codes_from_schema(categorical_schema::Dict{Symbol, CategoricalSchemaInfo}, data::NamedTuple) -> Dict{Symbol, Vector{Int}}

NEW: Extract level codes from schema information.
Replaces prepare_categorical_levels in the new architecture.
"""
function extract_level_codes_from_schema(categorical_schema::Dict{Symbol, CategoricalSchemaInfo}, data::NamedTuple)
    level_maps = Dict{Symbol, Vector{Int}}()
    
    for (col_name, schema_info) in categorical_schema
        level_maps[col_name] = schema_info.level_codes
    end
    
    return level_maps
end
