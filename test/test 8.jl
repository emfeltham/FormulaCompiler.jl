# test 8

println("=== Testing Complete Three-Phase Pipeline ===")

# Create comprehensive test data
Random.seed!(42)
n = 100
df = DataFrame(
    x = randn(n),
    y = randn(n),
    z = abs.(randn(n)) .+ 0.1,  # Positive for log
    w = randn(n),
    group = categorical(rand(["A", "B", "C"], n)),
    binary = categorical(rand(["Yes", "No"], n))
)

data = Tables.columntable(df)

# Test cases from simple to complex
test_formulas = [
    (@formula(y ~ 1), "Intercept only"),
    (@formula(y ~ x), "Simple continuous"),
    (@formula(y ~ group), "Simple categorical"),
    (@formula(y ~ x + group), "Mixed terms"),
    (@formula(y ~ x^2), "Power function"),
    (@formula(y ~ log(z)), "Log function"),
    (@formula(y ~ x + x^2 + log(z)), "Multiple functions"),
    (@formula(y ~ x * group), "Simple interaction"),
    (@formula(y ~ x^2 * log(z)), "Complex function interaction"),
    (@formula(y ~ x + x^2 + log(z) + group + w + x*group), "Kitchen sink"),
    (@formula(y ~ x*z*group), "Three-way interaction"),
    (@formula(y ~ (x>0) + log(z)*x), "Boolean and function interaction")
]

model = lm(formula, df)
        
println("=== Testing Complete Compilation Performance ===")

# Phase 1-3: Compilation (one-time cost)
println("\n1. Compilation Phase:")
compilation_time = @elapsed begin
    formula_val, output_width, column_names = compile_formula_complete(model)
end

println("   Compilation time: $(round(compilation_time * 1000, digits=2)) ms")

# Setup for runtime testing
row_vec = Vector{Float64}(undef, output_width)
n_rows = length(data[1])

println("\n2. Correctness Verification:")

# Test against model matrix
modelrow!(row_vec, formula_val, data, 1)
mm = modelmatrix(model)
expected = mm[1, :]

if isapprox(row_vec, expected, atol=1e-12)
    println("   ✅ Row 1 matches model matrix exactly")
else
    println("   ❌ Row 1 mismatch!")
    println("      Expected: $expected")
    println("      Got:      $row_vec")
    println("      Max diff: $(maximum(abs.(row_vec .- expected)))")
end