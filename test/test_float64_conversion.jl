# Test if Float64(val) conversion is causing the allocation
using FormulaCompiler
using DataFrames, GLM, Tables, StatsModels, BenchmarkTools

# Setup
n = 500
df = FormulaCompiler.make_test_data(; n)
data = Tables.columntable(df)

println("🎯 TESTING Float64 CONVERSION HYPOTHESIS")
println("=" ^ 45)

# Get a typical value from the data
val = data.x[1]
println("Test value: $val (type: $(typeof(val)))")

# Test the conversion patterns
println("\nTesting conversion patterns:")

# Test 1: Direct assignment (what we tested before)
buffer = [0.0]
test1 = @benchmark $buffer[1] = $val samples=1000
println("1. buffer[1] = val:          $(minimum(test1.memory)) bytes, $(minimum(test1.allocs)) allocs")

# Test 2: Float64 conversion assignment (what the code actually does)
test2 = @benchmark $buffer[1] = Float64($val) samples=1000
println("2. buffer[1] = Float64(val): $(minimum(test2.memory)) bytes, $(minimum(test2.allocs)) allocs")

# Test 3: Just the Float64 conversion
test3 = @benchmark Float64($val) samples=1000
println("3. Float64(val):             $(minimum(test3.memory)) bytes, $(minimum(test3.allocs)) allocs")

# Test 4: Type conversion pattern
result = 0.0
test4 = @benchmark $result = Float64($val) samples=1000
println("4. result = Float64(val):    $(minimum(test4.memory)) bytes, $(minimum(test4.allocs)) allocs")

# Test 5: Full pattern from execute_operation!
columns = (:x,)
positions = (1,)
buffer = [0.0]
test5 = @benchmark begin
    col = $columns[1]
    pos = $positions[1]
    val = FormulaCompiler.get_data_value_specialized($data, col, 1)
    $buffer[pos] = Float64(val)
end samples=1000
println("5. Full execute pattern:     $(minimum(test5.memory)) bytes, $(minimum(test5.allocs)) allocs")

# Test 6: Full pattern WITHOUT Float64 conversion
test6 = @benchmark begin
    col = $columns[1]
    pos = $positions[1]
    val = FormulaCompiler.get_data_value_specialized($data, col, 1)
    $buffer[pos] = val
end samples=1000
println("6. Pattern without Float64:  $(minimum(test6.memory)) bytes, $(minimum(test6.allocs)) allocs")

println("\n" * ("=" ^ 45))