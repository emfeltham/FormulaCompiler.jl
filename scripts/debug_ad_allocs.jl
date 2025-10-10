# Debug AD Allocations
# Minimal reproduction script to measure ForwardDiff allocations in derivative_modelrow!

using FormulaCompiler
using GLM, DataFrames, Tables
using BenchmarkTools

println("=" ^ 70)
println("AD Allocation Debug Script")
println("=" ^ 70)

# Minimal test case
n = 100
df = DataFrame(
    y = randn(n),
    x = randn(n),
    z = randn(n)
)

println("\nSetup:")
println("  Model: y ~ x + z (linear model)")
println("  Variables: [:x, :z]")
println("  Rows: $n")

# Compile formula
data = Tables.columntable(df)
model = lm(@formula(y ~ x + z), df)
compiled = compile_formula(model, data)
β = coef(model)

# Build evaluators
vars = [:x, :z]
de_ad = derivativeevaluator(:ad, compiled, data, vars)
de_fd = derivativeevaluator(:fd, compiled, data, vars)

# Pre-allocate buffers
J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(compiled), length(vars))

println("\nWarmup (100 iterations)...")
# Extensive warmup
for _ in 1:100
    derivative_modelrow!(J_ad, de_ad, 1)
    derivative_modelrow!(J_fd, de_fd, 1)
    marginal_effects_eta!(g, Gβ, de_ad, β, 1)
    marginal_effects_eta!(g, Gβ, de_fd, β, 1)
end

println("\n" * "=" ^ 70)
println("RESULTS: derivative_modelrow!")
println("=" ^ 70)

# Test AD derivative_modelrow!
b_ad_deriv = @benchmark derivative_modelrow!($J_ad, $de_ad, 1) samples=1000 evals=1
println("\nAD backend:")
println("  Memory: ", minimum(b_ad_deriv).memory, " bytes")
println("  Time:   ", round(minimum(b_ad_deriv).time, digits=1), " ns")
println("  Allocs: ", minimum(b_ad_deriv).allocs, " allocations")

# Test FD derivative_modelrow!
b_fd_deriv = @benchmark derivative_modelrow!($J_fd, $de_fd, 1) samples=1000 evals=1
println("\nFD backend:")
println("  Memory: ", minimum(b_fd_deriv).memory, " bytes")
println("  Time:   ", round(minimum(b_fd_deriv).time, digits=1), " ns")
println("  Allocs: ", minimum(b_fd_deriv).allocs, " allocations")

println("\n" * "=" ^ 70)
println("RESULTS: marginal_effects_eta!")
println("=" ^ 70)

# Test AD marginal_effects_eta!
b_ad_eta = @benchmark marginal_effects_eta!($g, $Gβ, $de_ad, $β, 1) samples=1000 evals=1
println("\nAD backend:")
println("  Memory: ", minimum(b_ad_eta).memory, " bytes")
println("  Time:   ", round(minimum(b_ad_eta).time, digits=1), " ns")
println("  Allocs: ", minimum(b_ad_eta).allocs, " allocations")

# Test FD marginal_effects_eta!
b_fd_eta = @benchmark marginal_effects_eta!($g, $Gβ, $de_fd, $β, 1) samples=1000 evals=1
println("\nFD backend:")
println("  Memory: ", minimum(b_fd_eta).memory, " bytes")
println("  Time:   ", round(minimum(b_fd_eta).time, digits=1), " ns")
println("  Allocs: ", minimum(b_fd_eta).allocs, " allocations")

println("\n" * "=" ^ 70)
println("ANALYSIS")
println("=" ^ 70)

ad_deriv_mem = minimum(b_ad_deriv).memory
fd_deriv_mem = minimum(b_fd_deriv).memory
ad_eta_mem = minimum(b_ad_eta).memory
fd_eta_mem = minimum(b_fd_eta).memory

if ad_deriv_mem > 0
    println("\n⚠️  AD derivative_modelrow! allocates $(ad_deriv_mem) bytes")
    println("   Target: 0 bytes (matching FD backend)")
else
    println("\n✅ AD derivative_modelrow! achieves zero allocations")
end

if ad_eta_mem > 0
    println("\n⚠️  AD marginal_effects_eta! allocates $(ad_eta_mem) bytes")
    println("   Target: 0 bytes (matching FD backend)")
else
    println("\n✅ AD marginal_effects_eta! achieves zero allocations")
end

println("\n" * "=" ^ 70)
println("NEXT STEPS")
println("=" ^ 70)
println("\n1. Run with allocation tracking:")
println("   julia --track-allocation=user --project=. scripts/debug_ad_allocs.jl")
println("\n2. Check .mem files in src/evaluation/derivatives/:")
println("   - automatic_diff.jl.mem")
println("   - types.jl.mem")
println("   - evaluator.jl.mem")
println("\n3. Look for lines with high allocation counts")
println("=" ^ 70)
