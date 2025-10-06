using ForwardDiff, BenchmarkTools

# Test if Dual arithmetic allocates
println("Testing ForwardDiff.Dual arithmetic allocations...")

# Create duals
d1 = ForwardDiff.Dual(3.0, (1.0, 0.0))
d2 = ForwardDiff.Dual(2.0, (0.0, 1.0))

# Test basic operations
println("\n=== Addition ===")
b = @benchmark $d1 + $d2
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

println("\n=== Multiplication ===")
b = @benchmark $d1 * $d2
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

println("\n=== Complex expression ===")
b = @benchmark ($d1 + $d2) * ($d1 - $d2)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

# Test with vector operations
println("\n=== Vector of Duals ===")
v1 = [ForwardDiff.Dual(Float64(i), (1.0, 0.0)) for i in 1:3]
v2 = [ForwardDiff.Dual(Float64(i), (0.0, 1.0)) for i in 1:3]

println("\nVector element-wise multiply:")
b = @benchmark $v1[1] * $v2[1]
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

# Test assignment to pre-allocated vector
output = similar(v1)
println("\nAssignment to pre-allocated vector:")
b = @benchmark $output[1] = $v1[1] * $v2[1]
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")
