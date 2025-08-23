# Test type stability of the continuous execution pattern
using FormulaCompiler
using DataFrames, GLM, Tables, StatsModels, BenchmarkTools
using InteractiveUtils

# Setup
n = 500
df = FormulaCompiler.make_test_data(; n)
data = Tables.columntable(df)

println("🎯 TESTING TYPE STABILITY HYPOTHESIS")
println("=" ^ 45)

# Recreate the exact types
columns = (:x,)::NTuple{1, Symbol}
positions = (1,)::NTuple{1, Int}
buffer = [0.0]

println("Data types:")
println("  columns: $(typeof(columns))")
println("  positions: $(typeof(positions))")
println("  data: $(typeof(data))")

# Function to test type stability with @code_warntype
function test_continuous_pattern(columns, positions, buffer, data)
    @inbounds for i in 1:1
        col = columns[i]
        pos = positions[i]
        val = FormulaCompiler.get_data_value_specialized(data, col, 1)
        buffer[pos] = Float64(val)
    end
end

println("\nType analysis:")
println("@code_warntype for test_continuous_pattern:")
@code_warntype test_continuous_pattern(columns, positions, buffer, data)

# Let's also test if making the function type-stable helps
function test_continuous_pattern_stable(
    columns::NTuple{N, Symbol}, 
    positions::NTuple{N, Int}, 
    buffer::Vector{Float64}, 
    data::NamedTuple
) where N
    @inbounds for i in 1:N
        col = columns[i]
        pos = positions[i]
        val = FormulaCompiler.get_data_value_specialized(data, col, 1)
        buffer[pos] = Float64(val)
    end
end

println("\n@code_warntype for type-stable version:")
@code_warntype test_continuous_pattern_stable(columns, positions, buffer, data)

# Test the allocation behavior
println("\nAllocation tests:")

# Original pattern
test1 = @benchmark test_continuous_pattern($columns, $positions, $buffer, $data) samples=1000
println("1. Original pattern:     $(minimum(test1.memory)) bytes, $(minimum(test1.allocs)) allocs")

# Type-stable pattern  
test2 = @benchmark test_continuous_pattern_stable($columns, $positions, $buffer, $data) samples=1000
println("2. Type-stable pattern:  $(minimum(test2.memory)) bytes, $(minimum(test2.allocs)) allocs")

# Test with type annotations in the benchmark
test3 = @benchmark begin
    cols::NTuple{1, Symbol} = $columns
    poss::NTuple{1, Int} = $positions
    @inbounds for i in 1:1
        col = cols[i]
        pos = poss[i]
        val = FormulaCompiler.get_data_value_specialized($data, col, 1)
        $buffer[pos] = Float64(val)
    end
end samples=1000
println("3. Annotated in benchmark: $(minimum(test3.memory)) bytes, $(minimum(test3.allocs)) allocs")

println("\n" * ("=" ^ 45))