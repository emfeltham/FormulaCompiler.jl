# Test unsafe mutation of Dual numbers for zero-allocation AD

using ForwardDiff, BenchmarkTools

println("=" ^ 70)
println("Testing Unsafe Dual Mutation for Zero Allocations")
println("=" ^ 70)

# Setup
nvars = 2
DualT = ForwardDiff.Dual{Nothing, Float64, nvars}
partials1 = ForwardDiff.Partials{nvars, Float64}((1.0, 0.0))
partials2 = ForwardDiff.Partials{nvars, Float64}((0.0, 1.0))

# Pre-allocate dual vector (one-time setup)
x_dual_vec = [
    DualT(0.0, partials1),  # Will mutate value field
    DualT(0.0, partials2)   # Will mutate value field
]

println("\nInitial x_dual_vec:")
println("  x_dual_vec[1] = ", x_dual_vec[1])
println("  x_dual_vec[2] = ", x_dual_vec[2])

# Test unsafe mutation
new_values = [1.5, 2.5]

println("\n" * "=" ^ 70)
println("Approach 1: Unsafe mutation via setfield!")
println("=" ^ 70)

function unsafe_mutate_duals!(x_dual_vec, new_values)
    for i in eachindex(x_dual_vec)
        # Unsafe mutation of immutable field
        # This works because Dual is a bitstype
        unsafe_store!(
            Ptr{Float64}(pointer_from_objref(x_dual_vec) + (i-1) * sizeof(DualT)),
            new_values[i]
        )
    end
    return nothing
end

# Warmup
for _ in 1:100
    unsafe_mutate_duals!(x_dual_vec, new_values)
end

println("\nAfter mutation:")
println("  x_dual_vec[1] = ", x_dual_vec[1])
println("  x_dual_vec[2] = ", x_dual_vec[2])

# Benchmark
b_unsafe = @benchmark unsafe_mutate_duals!($x_dual_vec, $new_values) samples=1000 evals=1
println("\nBenchmark results:")
println("  Memory: ", minimum(b_unsafe).memory, " bytes")
println("  Allocs: ", minimum(b_unsafe).allocs)
println("  Time:   ", round(minimum(b_unsafe).time, digits=1), " ns")

# Test that partials are preserved
println("\n" * "=" ^ 70)
println("Verification: Partials preserved?")
println("=" ^ 70)
println("  x_dual_vec[1].partials = ", ForwardDiff.partials(x_dual_vec[1]))
println("  x_dual_vec[2].partials = ", ForwardDiff.partials(x_dual_vec[2]))
println("  Expected:     (1.0, 0.0) and (0.0, 1.0)")

if ForwardDiff.partials(x_dual_vec[1]) == partials1 &&
   ForwardDiff.partials(x_dual_vec[2]) == partials2
    println("\n✅ Partials correctly preserved!")
else
    println("\n❌ Partials corrupted - unsafe mutation failed")
end

# Compare to reconstruction approach
println("\n" * "=" ^ 70)
println("Approach 2: Reconstruction (current approach)")
println("=" ^ 70)

x_dual_vec2 = [DualT(0.0, partials1), DualT(0.0, partials2)]

function reconstruct_duals!(x_dual_vec, new_values, partials1, partials2)
    x_dual_vec[1] = DualT(new_values[1], partials1)
    x_dual_vec[2] = DualT(new_values[2], partials2)
    return nothing
end

# Warmup
for _ in 1:100
    reconstruct_duals!(x_dual_vec2, new_values, partials1, partials2)
end

# Benchmark
b_reconstruct = @benchmark reconstruct_duals!($x_dual_vec2, $new_values, $partials1, $partials2) samples=1000 evals=1
println("\nBenchmark results:")
println("  Memory: ", minimum(b_reconstruct).memory, " bytes")
println("  Allocs: ", minimum(b_reconstruct).allocs)
println("  Time:   ", round(minimum(b_reconstruct).time, digits=1), " ns")

# Summary
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println("Unsafe mutation: ", minimum(b_unsafe).memory, " bytes, ", minimum(b_unsafe).allocs, " allocs")
println("Reconstruction:  ", minimum(b_reconstruct).memory, " bytes, ", minimum(b_reconstruct).allocs, " allocs")

if minimum(b_unsafe).memory == 0
    println("\n✅ SUCCESS: Unsafe mutation achieves ZERO allocations!")
    println("   This is the solution for zero-alloc AD backend.")
else
    println("\n⚠️  Unsafe mutation still allocates - need different approach")
end
