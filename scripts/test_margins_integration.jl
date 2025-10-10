#!/usr/bin/env julia
# Test FormulaCompiler AD zero-allocations with Margins.jl-style usage

using FormulaCompiler
using GLM
using DataFrames
using BenchmarkTools
using Tables

println("=" ^ 70)
println("TESTING: FormulaCompiler AD zero-allocations with Margins.jl workflow")
println("=" ^ 70)
println()

# Create test data (similar to Margins.jl tests)
n = 1000
data_df = DataFrame(
    x1 = randn(n),
    x2 = randn(n),
    x3 = rand([0, 1], n),
    y = randn(n)
)

# Fit model
model = lm(@formula(y ~ x1 + x2 + x3), data_df)

# Convert to columntable (what Margins.jl does)
data_nt = Tables.columntable(data_df)

# Compile formula
compiled = compile_formula(model, data_nt)

# Build AD evaluator for continuous variables
vars = [:x1, :x2]
de_ad = derivativeevaluator(:ad, compiled, data_nt, vars)
de_fd = derivativeevaluator(:fd, compiled, data_nt, vars)

# Pre-allocate buffers
J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
g_ad = Vector{Float64}(undef, length(vars))
g_fd = Vector{Float64}(undef, length(vars))
Gβ_ad = Matrix{Float64}(undef, length(compiled), length(vars))
Gβ_fd = Matrix{Float64}(undef, length(compiled), length(vars))
β = coef(model)

# Warmup
derivative_modelrow!(J_ad, de_ad, 1)
derivative_modelrow!(J_fd, de_fd, 1)
marginal_effects_eta!(g_ad, Gβ_ad, de_ad, β, 1)
marginal_effects_eta!(g_fd, Gβ_fd, de_fd, β, 1)

println("Testing derivative_modelrow! (single row):")
println("-" ^ 70)

# Test AD backend
bench_ad = @benchmark derivative_modelrow!($J_ad, $de_ad, 1) samples=1000
println("AD Backend:")
println("  Memory: $(minimum(bench_ad).memory) bytes")
println("  Allocs: $(minimum(bench_ad).allocs)")
println("  Time:   $(minimum(bench_ad).time) ns")
println()

# Test FD backend
bench_fd = @benchmark derivative_modelrow!($J_fd, $de_fd, 1) samples=1000
println("FD Backend:")
println("  Memory: $(minimum(bench_fd).memory) bytes")
println("  Allocs: $(minimum(bench_fd).allocs)")
println("  Time:   $(minimum(bench_fd).time) ns")
println()

println("Testing marginal_effects_eta! (single row):")
println("-" ^ 70)

# Test AD backend
bench_me_ad = @benchmark marginal_effects_eta!($g_ad, $Gβ_ad, $de_ad, $β, 1) samples=1000
println("AD Backend:")
println("  Memory: $(minimum(bench_me_ad).memory) bytes")
println("  Allocs: $(minimum(bench_me_ad).allocs)")
println("  Time:   $(minimum(bench_me_ad).time) ns")
println()

# Test FD backend
bench_me_fd = @benchmark marginal_effects_eta!($g_fd, $Gβ_fd, $de_fd, $β, 1) samples=1000
println("FD Backend:")
println("  Memory: $(minimum(bench_me_fd).memory) bytes")
println("  Allocs: $(minimum(bench_me_fd).allocs)")
println("  Time:   $(minimum(bench_me_fd).time) ns")
println()

# Test loop performance (what Margins.jl does)
println("Testing loop performance (1000 rows, like Margins.jl):")
println("-" ^ 70)

function test_loop_ad(J, de, rows)
    for row in rows
        derivative_modelrow!(J, de, row)
    end
end

function test_loop_fd(J, de, rows)
    for row in rows
        derivative_modelrow!(J, de, row)
    end
end

rows = 1:n

# Warmup
test_loop_ad(J_ad, de_ad, rows)
test_loop_fd(J_fd, de_fd, rows)

bench_loop_ad = @benchmark test_loop_ad($J_ad, $de_ad, $rows) samples=100
println("AD Loop (1000 rows):")
println("  Total memory: $(minimum(bench_loop_ad).memory) bytes")
println("  Total allocs: $(minimum(bench_loop_ad).allocs)")
println("  Per-row memory: $(minimum(bench_loop_ad).memory / n) bytes")
println("  Per-row allocs: $(minimum(bench_loop_ad).allocs / n)")
println("  Total time: $(minimum(bench_loop_ad).time / 1e6) ms")
println()

bench_loop_fd = @benchmark test_loop_fd($J_fd, $de_fd, $rows) samples=100
println("FD Loop (1000 rows):")
println("  Total memory: $(minimum(bench_loop_fd).memory) bytes")
println("  Total allocs: $(minimum(bench_loop_fd).allocs)")
println("  Per-row memory: $(minimum(bench_loop_fd).memory / n) bytes")
println("  Per-row allocs: $(minimum(bench_loop_fd).allocs / n)")
println("  Total time: $(minimum(bench_loop_fd).time / 1e6) ms")
println()

println("=" ^ 70)
if minimum(bench_ad).memory == 0 && minimum(bench_fd).memory == 0
    println("✅ SUCCESS: FormulaCompiler achieves ZERO ALLOCATIONS for both backends")
    println("✅ Per-row AD allocs: $(minimum(bench_loop_ad).allocs / n)")
    println("✅ Per-row FD allocs: $(minimum(bench_loop_fd).allocs / n)")
    println()
    println("Margins.jl allocations must be coming from Margins.jl infrastructure,")
    println("NOT from FormulaCompiler derivative primitives.")
else
    println("❌ FAILED: FormulaCompiler still has allocations")
end
println("=" ^ 70)
