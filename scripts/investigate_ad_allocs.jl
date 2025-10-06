# Investigate what causes AD allocations in complex formula

using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays, BenchmarkTools

println("Testing what causes AD allocations...")
println("=" ^ 60)

# Test 1: Many categorical levels
println("\nTest 1: Many categorical levels (10 levels)")
n = 300
df1 = DataFrame(
    y = randn(n),
    x = randn(n),
    cat = categorical(rand([string(Char(65+i)) for i in 0:9], n))  # 10 levels
)
model1 = lm(@formula(y ~ x + x&cat), df1)
data1 = Tables.columntable(df1)
compiled1 = compile_formula(model1, data1)
de_ad1 = derivativeevaluator(:ad, compiled1, data1, [:x])
J1 = Matrix{Float64}(undef, length(compiled1), 1)
derivative_modelrow!(J1, de_ad1, 1)
b1 = @benchmark derivative_modelrow!($J1, $de_ad1, 1) samples=400
println("  $(length(compiled1)) terms: $(minimum(b1).allocs) allocs, $(minimum(b1).memory) bytes")

# Test 2: Very large formula (256 terms like complex_example)
println("\nTest 2: Formula with ~250 terms (many categorical interactions)")
df2 = DataFrame(
    y = randn(n),
    x1 = randn(n),
    x2 = randn(n),
    x3 = randn(n),
    cat1 = categorical(rand(["A", "B", "C", "D", "E"], n)),
    cat2 = categorical(rand(["X", "Y", "Z"], n)),
    cat3 = categorical(rand(["P", "Q"], n))
)
# Create many interactions to get ~250 terms
model2 = lm(@formula(y ~ x1 + x2 + x3 +
                         x1&cat1 + x2&cat1 + x3&cat1 +
                         x1&cat2 + x2&cat2 + x3&cat2 +
                         x1&cat3 + x2&cat3 + x3&cat3 +
                         cat1&cat2 + cat1&cat3 + cat2&cat3 +
                         x1&cat1&cat2 + x2&cat1&cat2 + x3&cat1&cat2), df2)
data2 = Tables.columntable(df2)
compiled2 = compile_formula(model2, data2)
de_ad2 = derivativeevaluator(:ad, compiled2, data2, [:x1, :x2])
J2 = Matrix{Float64}(undef, length(compiled2), 2)
derivative_modelrow!(J2, de_ad2, 1)
b2 = @benchmark derivative_modelrow!($J2, $de_ad2, 1) samples=400
println("  $(length(compiled2)) terms: $(minimum(b2).allocs) allocs, $(minimum(b2).memory) bytes")

# Test 3: Simple formula for comparison (should be 0 allocs)
println("\nTest 3: Simple formula for comparison")
df3 = DataFrame(
    y = randn(n),
    x = randn(n),
    z = randn(n),
    cat = categorical(rand(["A", "B"], n))
)
model3 = lm(@formula(y ~ x + z + x&cat), df3)
data3 = Tables.columntable(df3)
compiled3 = compile_formula(model3, data3)
de_ad3 = derivativeevaluator(:ad, compiled3, data3, [:x, :z])
J3 = Matrix{Float64}(undef, length(compiled3), 2)
derivative_modelrow!(J3, de_ad3, 1)
b3 = @benchmark derivative_modelrow!($J3, $de_ad3, 1) samples=400
println("  $(length(compiled3)) terms: $(minimum(b3).allocs) allocs, $(minimum(b3).memory) bytes")
