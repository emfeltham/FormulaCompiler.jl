# Test setfield! for mutating Dual numbers

using ForwardDiff, BenchmarkTools

println("=" ^ 70)
println("Testing setfield! for Dual Mutation")
println("=" ^ 70)

# Setup
nvars = 2
DualT = ForwardDiff.Dual{Nothing, Float64, nvars}
partials1 = ForwardDiff.Partials{nvars, Float64}((1.0, 0.0))
partials2 = ForwardDiff.Partials{nvars, Float64}((0.0, 1.0))

# Pre-allocate dual vector
x_dual_vec = [
    DualT(0.0, partials1),
    DualT(0.0, partials2)
]

println("\nInitial:")
println("  x_dual_vec[1] = ", x_dual_vec[1])
println("  x_dual_vec[2] = ", x_dual_vec[2])

# Try setfield! (works for immutable types)
new_value = 1.5

println("\n" * "=" ^ 70)
println("Approach: Using Base.setfield! (immutable field mutation)")
println("=" ^ 70)

try
    # Attempt 1: Direct setfield! on element
    Base.setfield!(x_dual_vec[1], :value, new_value)
    println("✅ setfield! succeeded")
    println("  x_dual_vec[1] = ", x_dual_vec[1])
catch e
    println("❌ setfield! failed: ", e)
end

# Alternative: Pre-compute all needed dual values
println("\n" * "=" ^ 70)
println("Alternative: Pre-computed Dual Cache")
println("=" ^ 70)

# Instead of mutating, maintain a cache of commonly-used dual values
# This avoids reconstruction in the hot loop

# For a given variable, we need duals with different values but same partials
# We can't avoid construction, but we can batch it

println("\nProblem: Dual is immutable, cannot be mutated safely")
println("Conclusion: Must either:")
println("  1. Accept the ~64 bytes per construction")
println("  2. Use a completely different approach (no Dual at all)")
println("  3. Implement custom dual-number type that IS mutable")

println("\n" * "=" ^ 70)
println("RECOMMENDATION")
println("=" ^ 70)
println("Since Dual is immutable and unsafe mutation crashes,")
println("the best path forward is:")
println("")
println("Option A: Custom mutable Dual type")
println("  - Define MutableDual{T,N} with mutable fields")
println("  - Implement ForwardDiff-compatible arithmetic")
println("  - Use in place of ForwardDiff.Dual")
println("")
println("Option B: Eliminate Dual entirely")
println("  - Represent value + partials as separate arrays")
println("  - Manually propagate derivatives")
println("  - Avoid all Dual construction")
println("")
println("Option A is safer and maintains ForwardDiff compatibility.")
println("Option B is faster but requires custom derivative rules.")
