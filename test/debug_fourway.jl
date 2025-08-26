# Debug four-way interaction issue
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays
using Test

println("="^70)
println("DEBUGGING FOUR-WAY INTERACTION")
println("="^70)

# Create minimal test data
n = 20
df = DataFrame(
    x = randn(n),
    y = randn(n),
    group3 = categorical(repeat(["A", "B", "C"], outer=7)[1:n]),  # 3 levels
    group4 = categorical(repeat(["P", "Q", "R", "S"], outer=5)[1:n]),  # 4 levels
    continuous_response = randn(n)
)
data = Tables.columntable(df)

println("\nData structure:")
println("- n = $n")
println("- group3 levels: ", levels(df.group3))
println("- group4 levels: ", levels(df.group4))

# Test four-way interaction
formula = @formula(continuous_response ~ x * y * group3 * group4)
println("\nFormula: ", formula)

# Fit model and get expected matrix
model = lm(formula, df)
expected_matrix = modelmatrix(model)

println("\nExpected model matrix:")
println("- Size: ", size(expected_matrix))
println("- Columns: ", size(expected_matrix, 2))

# Compile formula
println("\nCompiling formula...")
compiled = compile_formula(model, data)
println("- Compiled output size: ", length(compiled))

# Test correctness on first few rows
output = Vector{Float64}(undef, length(compiled))

println("\nTesting correctness:")
for row in 1:min(5, n)
    compiled(output, data, row)
    expected = expected_matrix[row, :]
    
    if length(output) != length(expected)
        println("❌ Row $row: Size mismatch!")
        println("   Output size: $(length(output))")
        println("   Expected size: $(length(expected))")
        break
    elseif !isapprox(output, expected, rtol=1e-10)
        println("❌ Row $row: Values differ!")
        println("   Max diff: $(maximum(abs.(output .- expected)))")
        # Show which positions differ
        for (i, (o, e)) in enumerate(zip(output, expected))
            if !isapprox(o, e, rtol=1e-10)
                println("   Position $i: output=$o, expected=$e, diff=$(abs(o-e))")
            end
        end
    else
        println("✅ Row $row: Correct")
    end
end

# Let's examine the formula expansion
println("\n" * "="^70)
println("FORMULA STRUCTURE ANALYSIS")
println("="^70)

# Calculate expected dimensions
n_x = 1  # continuous
n_y = 1  # continuous  
n_group3 = 2  # 3 levels = 2 contrasts (dropping reference)
n_group4 = 3  # 4 levels = 3 contrasts (dropping reference)

println("\nComponent dimensions:")
println("- x: $n_x")
println("- y: $n_y")
println("- group3 contrasts: $n_group3")
println("- group4 contrasts: $n_group4")

# Four-way interaction expansion
# Full expansion: x * y * group3 * group4
# This includes all lower-order terms:
# - Main effects: x, y, group3, group4
# - 2-way: x*y, x*group3, x*group4, y*group3, y*group4, group3*group4
# - 3-way: x*y*group3, x*y*group4, x*group3*group4, y*group3*group4
# - 4-way: x*y*group3*group4

println("\nExpected term counts:")
println("- Intercept: 1")
println("- Main effects: x(1) + y(1) + group3(2) + group4(3) = 7")
println("- 2-way interactions:")
println("  - x*y: 1×1 = 1")
println("  - x*group3: 1×2 = 2")
println("  - x*group4: 1×3 = 3")
println("  - y*group3: 1×2 = 2")
println("  - y*group4: 1×3 = 3")
println("  - group3*group4: 2×3 = 6")
println("  - Total 2-way: 17")
println("- 3-way interactions:")
println("  - x*y*group3: 1×1×2 = 2")
println("  - x*y*group4: 1×1×3 = 3")
println("  - x*group3*group4: 1×2×3 = 6")
println("  - y*group3*group4: 1×2×3 = 6")
println("  - Total 3-way: 17")
println("- 4-way interaction:")
println("  - x*y*group3*group4: 1×1×2×3 = 6")

total_expected = 1 + 7 + 17 + 17 + 6
println("\nTotal expected columns: $total_expected")
println("Actual model matrix columns: $(size(expected_matrix, 2))")
println("Compiled output size: $(length(compiled))")

if size(expected_matrix, 2) != length(compiled)
    println("\n❌ MISMATCH DETECTED!")
    println("Missing columns: $(size(expected_matrix, 2) - length(compiled))")
end