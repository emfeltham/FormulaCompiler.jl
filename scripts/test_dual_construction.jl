using ForwardDiff, BenchmarkTools

# Test Dual construction
println("Testing ForwardDiff.Dual construction...")

# Test 1: Construction with tuple
p = (1.0, 0.0)
println("\n=== Construction with NTuple ===")
b = @benchmark ForwardDiff.Dual(3.0, $p)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

# Test 2: Construction with Partials
parts = ForwardDiff.Partials{2,Float64}((1.0, 0.0))
println("\n=== Construction with Partials ===")
b = @benchmark ForwardDiff.Dual(3.0, $parts)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

# Test 3: Construction via typeof
d = ForwardDiff.Dual{Nothing, Float64, 2}(2.0, parts)
println("\n=== Construction via typeof ===")
b = @benchmark typeof($d)(3.0, $parts)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

# Test 4: Assignment to vector
vec = Vector{typeof(d)}(undef, 2)
vec[1] = d
vec[2] = d
println("\n=== Assignment to pre-allocated vector ===")
b = @benchmark $vec[1] = typeof($d)(5.0, $parts)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

# Test 5: Creation in a loop (mimicking our code)
partials_vec = [ForwardDiff.Partials{2,Float64}((i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0)) for i in 1:2]
x_dual_vec = [ForwardDiff.Dual{Nothing, Float64, 2}(0.0, partials_vec[i]) for i in 1:2]

println("\n=== Loop construction (like our code) ===")
b = @benchmark begin
    for i in 1:2
        base_val = Float64(i * 2.0)
        $x_dual_vec[i] = typeof($x_dual_vec[i])(base_val, $partials_vec[i])
    end
end
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")
