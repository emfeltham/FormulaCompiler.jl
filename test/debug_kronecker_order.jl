# Debug Kronecker product ordering for four-way interactions
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays
using FormulaCompiler: make_test_data, decompose_formula, CompilationContext, compute_all_interaction_combinations
using Random

Random.seed!(08540)

# Use the same test data
n = 500
df = make_test_data(; n)
data = Tables.columntable(df)

# Get the four-way formula
formula = @formula(continuous_response ~ x * y * group3 * group4)
model = lm(formula, df)

println("="^70)
println("DEBUGGING KRONECKER ORDERING FOR 4-WAY")
println("="^70)

# Check what row 250 produces
row = 250
println("\nRow $row data:")
println("  x = $(df.x[row])")
println("  y = $(df.y[row])")
println("  group3 = $(df.group3[row]) (level $(levelcode(df.group3[row])))")
println("  group4 = $(df.group4[row]) (level $(levelcode(df.group4[row])))")

# Trace through our Kronecker product logic
println("\n" * "="^70)
println("Manual Kronecker Product Trace")
println("="^70)

# For row 250: group3=B (level 2), group4=Y (level 3)
# group3 contrasts: B→[1,0], C→[0,1] (positions 1-2)
# group4 contrasts: X→[1,0,0], Y→[0,1,0], Z→[0,0,1] (positions 1-3)

g3_contrasts = [1.0, 0.0]  # B level
g4_contrasts = [0.0, 1.0, 0.0]  # Y level

println("\ngroup3 (B) contrasts: $g3_contrasts")
println("group4 (Y) contrasts: $g4_contrasts")

# Test how compute_all_interaction_combinations works
component_positions = [[1, 2], [1, 2, 3]]  # group3 has 2 positions, group4 has 3

result = compute_all_interaction_combinations(component_positions)
println("\nKronecker expansion of positions:")
for (i, combo) in enumerate(result)
    println("  Combo $i: $combo")
end

# Expected mapping for group3 × group4:
println("\nExpected categorical interaction mapping:")
println("  [1,1] → group3:B & group4:X")
println("  [2,1] → group3:C & group4:X")
println("  [1,2] → group3:B & group4:Y")  # This should be active for row 250
println("  [2,2] → group3:C & group4:Y")
println("  [1,3] → group3:B & group4:Z")
println("  [2,3] → group3:C & group4:Z")

# Now let's check what StatsModels produces
mm = modelmatrix(model)
println("\n" * "="^70)
println("StatsModels Output for Row $row")
println("="^70)

# Find the positions for group3&group4 interaction terms
coef_names = coefnames(model)
for (i, name) in enumerate(coef_names)
    if contains(name, "group3") && contains(name, "group4") && !contains(name, "x") && !contains(name, "y")
        println("  Column $i ($name): $(mm[row, i])")
    end
end

# Now check what we produce
compiled = compile_formula(model, data)
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, row)

println("\n" * "="^70)
println("Our Output for Row $row")
println("="^70)

for (i, name) in enumerate(coef_names)
    if contains(name, "group3") && contains(name, "group4") && !contains(name, "x") && !contains(name, "y")
        println("  Column $i ($name): $(output[i])")
        if !isapprox(output[i], mm[row, i], rtol=1e-10)
            println("    ❌ MISMATCH! Expected $(mm[row, i])")
        end
    end
end

# Let's also check the 4-way interaction terms
println("\n" * "="^70)
println("Four-way interaction terms (x & y & group3 & group4)")
println("="^70)

for (i, name) in enumerate(coef_names)
    if contains(name, "x") && contains(name, "y") && contains(name, "group3") && contains(name, "group4")
        println("  Column $i ($name):")
        println("    StatsModels: $(mm[row, i])")
        println("    Our output: $(output[i])")
        if !isapprox(output[i], mm[row, i], rtol=1e-10)
            println("    ❌ MISMATCH!")
        end
    end
end