using FormulaCompiler, GLM, DataFrames, Tables, BenchmarkTools

println("="^70)
println("VERIFYING ZERO ALLOCATIONS FOR AD BACKEND")
println("="^70)

# Simple model
df = DataFrame(y = randn(1000), x = randn(1000), z = randn(1000))
model = lm(@formula(y ~ x + z), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build evaluators
vars = [:x, :z]
println("\nBuilding evaluators...")
de_ad = FormulaCompiler.derivativeevaluator_ad(compiled, data, vars)
de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data, vars)

# Allocate buffers
J = Matrix{Float64}(undef, length(compiled), length(vars))
g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(compiled), length(vars))
β = coef(model)

# Warmup
println("Warming up...")
for i in 1:1000
    FormulaCompiler.derivative_modelrow!(J, de_ad, i)
    FormulaCompiler.derivative_modelrow!(J, de_fd, i)
end

println("\n" * repeat("=", 70))
println("RESULTS")
println(repeat("=", 70))

# Test derivative_modelrow!
println("\n=== derivative_modelrow! ===")
b_ad = @benchmark FormulaCompiler.derivative_modelrow!($J, $de_ad, 500)
b_fd = @benchmark FormulaCompiler.derivative_modelrow!($J, $de_fd, 500)

println("AD Backend:")
println("  Memory: $(minimum(b_ad).memory) bytes")
println("  Allocs: $(minimum(b_ad).allocs)")
println("  Time:   $(round(minimum(b_ad).time, digits=1)) ns")

println("\nFD Backend:")
println("  Memory: $(minimum(b_fd).memory) bytes")
println("  Allocs: $(minimum(b_fd).allocs)")
println("  Time:   $(round(minimum(b_fd).time, digits=1)) ns")

# Test marginal_effects_eta!
println("\n=== marginal_effects_eta! ===")
b_ad = @benchmark FormulaCompiler.marginal_effects_eta!($g, $Gβ, $de_ad, $β, 500)
b_fd = @benchmark FormulaCompiler.marginal_effects_eta!($g, $Gβ, $de_fd, $β, 500)

println("AD Backend:")
println("  Memory: $(minimum(b_ad).memory) bytes")
println("  Allocs: $(minimum(b_ad).allocs)")
println("  Time:   $(round(minimum(b_ad).time, digits=1)) ns")

println("\nFD Backend:")
println("  Memory: $(minimum(b_fd).memory) bytes")
println("  Allocs: $(minimum(b_fd).allocs)")
println("  Time:   $(round(minimum(b_fd).time, digits=1)) ns")

# Success check
println("\n" * repeat("=", 70))
if minimum(b_ad).memory == 0 && minimum(b_fd).memory == 0
    println("✅ SUCCESS: Both backends achieve ZERO ALLOCATIONS!")
    println(repeat("=", 70))
    exit(0)
else
    println("❌ FAILURE: Allocations detected")
    println(repeat("=", 70))
    exit(1)
end
