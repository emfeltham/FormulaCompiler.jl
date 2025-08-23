# Test if the @inbounds loop pattern is causing the allocation
using FormulaCompiler
using DataFrames, GLM, Tables, StatsModels, BenchmarkTools

# Setup
n = 500
df = FormulaCompiler.make_test_data(; n)
data = Tables.columntable(df)

println("🎯 TESTING LOOP PATTERN HYPOTHESIS")
println("=" ^ 40)

# Recreate the exact data structures from ContinuousData
columns = (:x,)  # NTuple{1, Symbol}
positions = (1,) # NTuple{1, Int}
buffer = [0.0]

println("Testing different loop patterns:")

# Test 1: Direct indexing (what we tested before)
test1 = @benchmark begin
    col = $columns[1]
    pos = $positions[1]
    val = FormulaCompiler.get_data_value_specialized($data, col, 1)
    $buffer[pos] = Float64(val)
end samples=1000
println("1. Direct indexing:           $(minimum(test1.memory)) bytes, $(minimum(test1.allocs)) allocs")

# Test 2: @inbounds loop (exact execute_operation! pattern)
test2 = @benchmark begin
    @inbounds for i in 1:1
        col = $columns[i]
        pos = $positions[i]
        val = FormulaCompiler.get_data_value_specialized($data, col, 1)
        $buffer[pos] = Float64(val)
    end
end samples=1000
println("2. @inbounds for loop:        $(minimum(test2.memory)) bytes, $(minimum(test2.allocs)) allocs")

# Test 3: Regular loop
test3 = @benchmark begin
    for i in 1:1
        col = $columns[i]
        pos = $positions[i]
        val = FormulaCompiler.get_data_value_specialized($data, col, 1)
        $buffer[pos] = Float64(val)
    end
end samples=1000
println("3. Regular for loop:          $(minimum(test3.memory)) bytes, $(minimum(test3.allocs)) allocs")

# Test 4: Unrolled (no loop)
test4 = @benchmark begin
    col = $columns[1]
    pos = $positions[1]
    val = FormulaCompiler.get_data_value_specialized($data, col, 1)
    $buffer[pos] = Float64(val)
end samples=1000
println("4. Unrolled (no loop):        $(minimum(test4.memory)) bytes, $(minimum(test4.allocs)) allocs")

# Test 5: Check if it's the tuple unpacking in the loop
test5 = @benchmark begin
    @inbounds for i in 1:1
        col = $columns[i]
    end
end samples=1000
println("5. Just tuple access in loop: $(minimum(test5.memory)) bytes, $(minimum(test5.allocs)) allocs")

# Test 6: Check if it's the iteration itself
test6 = @benchmark begin
    @inbounds for i in 1:1
        # do nothing
    end
end samples=1000
println("6. Empty @inbounds loop:      $(minimum(test6.memory)) bytes, $(minimum(test6.allocs)) allocs")

# Test 7: Check multiple iterations
test7 = @benchmark begin
    @inbounds for i in 1:1
        val = FormulaCompiler.get_data_value_specialized($data, :x, 1)
    end
end samples=1000
println("7. Loop with data access:     $(minimum(test7.memory)) bytes, $(minimum(test7.allocs)) allocs")

# Test 8: Check if it's the specific function call pattern
test8 = @benchmark begin
    @inbounds for i in 1:1
        col_sym = :x
        val = FormulaCompiler.get_data_value_specialized($data, col_sym, 1)
        $buffer[1] = Float64(val)
    end
end samples=1000
println("8. Loop with literal symbols: $(minimum(test8.memory)) bytes, $(minimum(test8.allocs)) allocs")

println("\n" * ("=" ^ 40))