# validation_test.jl
# Simple diagnostic to understand the interaction structure

using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, Random, BenchmarkTools
using StatsModels

Random.seed!(12345)

n = 100
df = DataFrame(x = randn(n), y = randn(n), response = randn(n))
data = Tables.columntable(df)

formula = @formula(response ~ x * y)
model = fit(LinearModel, formula, df)
compiled = FormulaCompiler.compile_formula_specialized(model, data)
output = Vector{Float64}(undef, length(compiled))

# Warmup
for _ in 1:10
    compiled(output, data, 1)
end

println("="^50)
println("DIAGNOSTIC: Understanding x * y structure")
println("="^50)

# Current baseline
baseline = @allocated begin
    for i in 1:100
        compiled(output, data, i)
    end
end

println("Current: $(baseline / 100) bytes per call")
println("Output width: $(length(compiled))")

# Examine the interaction structure
interactions = compiled.data.interactions
println("\nInteraction analysis:")
println("Number of interactions: $(length(interactions))")

if !isempty(interactions)
    interaction = interactions[1]
    println("\nInteraction details:")
    println("  Type: $(typeof(interaction))")
    println("  Components: $(length(interaction.components))")
    println("  Positions: $(interaction.positions)")
    println("  Scratch map: $(interaction.component_scratch_map)")
    println("  Internal scratch map: $(interaction.component_internal_scratch_map)")
    println("  Kronecker pattern: $(interaction.kronecker_pattern)")
    println("  Total scratch needed: $(interaction.total_scratch_needed)")
    
    for (i, comp) in enumerate(interaction.components)
        println("  Component $i:")
        println("    Type: $(typeof(comp))")
        if comp isa FormulaCompiler.ContinuousEvaluator
            println("    Column: $(comp.column)")
            println("    Position: $(comp.position)")
        end
    end
end

# Check what the compiled formula actually produces
println("\nActual execution result:")
fill!(output, NaN)
compiled(output, data, 1)
println("Output: $(output)")

println("btime")
@btime compiled(output, data, 1)

# Check individual parts
println("\nBreaking down execution:")

# Constants
println("Constants:")
FormulaCompiler.execute_complete_constant_operations!(compiled.data.constants, output, data, 1)
println("  After constants: $(output)")

# Continuous  
FormulaCompiler.execute_complete_continuous_operations!(compiled.data.continuous, output, data, 1)
println("  After continuous: $(output)")

# Interactions
scratch = compiled.data.interaction_scratch
if !isempty(interactions)
    FormulaCompiler.execute_interaction_operations!(interactions, scratch, output, data, 1)
    println("  After interactions: $(output)")
    println("  Scratch after interactions: $(scratch)")
end

# Simple test: What's the minimum allocation we can get?
println("\nMinimal allocation tests:")

# Test 1: Just data access
test1 = @allocated begin
    for i in 1:100
        x_val = data.x[i % length(data.x) + 1]
        y_val = data.y[i % length(data.y) + 1]
    end
end
println("Direct data access: $(test1 / 100) bytes")

# Test 2: Data access + basic math
test2 = @allocated begin
    for i in 1:100
        x_val = data.x[i % length(data.x) + 1]  
        y_val = data.y[i % length(data.y) + 1]
        result = x_val * y_val
    end
end
println("Data access + multiply: $(test2 / 100) bytes")

# Test 3: Data access + array writes
test3 = @allocated begin
    temp_output = Vector{Float64}(undef, length(output))
    for i in 1:100
        x_val = data.x[i % length(data.x) + 1]
        y_val = data.y[i % length(data.y) + 1] 
        temp_output[1] = x_val * y_val
    end
end
println("Data access + array write: $(test3 / 100) bytes")

# Test 4: Using get_data_value_specialized
test4 = @allocated begin
    for i in 1:100
        row_idx = i % length(data.x) + 1
        x_val = FormulaCompiler.get_data_value_specialized(data, :x, row_idx)
        y_val = FormulaCompiler.get_data_value_specialized(data, :y, row_idx)
    end
end
println("get_data_value_specialized: $(test4 / 100) bytes")

# Test 5: Recreate what the interaction should do (correctly this time)
if !isempty(interactions)
    interaction = interactions[1]
    
    test5 = @allocated begin
        for call in 1:100
            row_idx = call % length(data.x) + 1
            
            # Load components into scratch (correctly)
            for i in 1:length(interaction.components)
                component = interaction.components[i]
                output_range = interaction.component_scratch_map[i]
                
                if component isa FormulaCompiler.ContinuousEvaluator && !isempty(output_range)
                    val = FormulaCompiler.get_data_value_specialized(data, component.column, row_idx)
                    scratch[first(output_range)] = Float64(val)
                end
            end
            
            # Apply Kronecker pattern (correctly)
            for result_idx in 1:length(interaction.kronecker_pattern)
                if result_idx <= length(interaction.positions)
                    pattern = interaction.kronecker_pattern[result_idx]
                    product = 1.0
                    
                    for comp_idx in 1:length(pattern)
                        pattern_val = pattern[comp_idx]
                        comp_range = interaction.component_scratch_map[comp_idx]
                        value_pos = first(comp_range) + pattern_val - 1
                        product *= scratch[value_pos]
                    end
                    
                    output_pos = interaction.positions[result_idx]
                    output[output_pos] = product
                end
            end
        end
    end
    
    println("Manual interaction (correct): $(test5 / 100) bytes")
end

println("\n" * "="^50)
println("SUMMARY")
println("="^50)
println("Current execution: $(baseline / 100) bytes")
println("Minimum theoretically possible: $(min(test1, test2, test3, test4) / 100) bytes")

if min(test1, test2, test3, test4) == 0
    println("🎯 Zero allocations should be possible")
    println("📋 The $(baseline / 100) bytes must be coming from:")
    println("   - Function call overhead")
    println("   - Type instability") 
    println("   - Hidden allocations in FormulaCompiler code")
else
    println("⚠️  Even basic operations allocate")
    println("📋 Problem might be:")
    println("   - Julia version/setup issue")
    println("   - Measurement methodology")
    println("   - Fundamental limitation")
end
