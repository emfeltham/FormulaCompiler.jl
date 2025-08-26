# Debug categorical level extraction
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays
using FormulaCompiler: make_test_data, compile_formula, extract_level_code_zero_alloc
using Random

Random.seed!(08540)

# Use the same test data
n = 500
df = make_test_data(; n)
data = Tables.columntable(df)

row = 250

println("Checking level extraction for row $row:")
println("="^60)

# Check group3
g3_val = df.group3[row]
g3_level = levelcode(df.group3[row])
println("\ngroup3:")
println("  Value: $g3_val")
println("  Level code: $g3_level")
println("  All levels: $(levels(df.group3))")

# Check group4  
g4_val = df.group4[row]
g4_level = levelcode(df.group4[row])
println("\ngroup4:")
println("  Value: $g4_val")
println("  Level code: $g4_level")
println("  All levels: $(levels(df.group4))")

# Test extract_level_code_zero_alloc
g3_extracted = extract_level_code_zero_alloc(data.group3, row)
g4_extracted = extract_level_code_zero_alloc(data.group4, row)

println("\nExtracted levels:")
println("  group3: $g3_extracted (should be $g3_level)")
println("  group4: $g4_extracted (should be $g4_level)")

# Check contrast matrices
formula = @formula(continuous_response ~ group3 * group4)
model = lm(formula, df)

# The issue might be in how we map levels to contrast positions
println("\n" * "="^60)
println("Contrast matrix mapping:")
println("="^60)

# For group3 with levels A, B, C and DummyCoding:
# Level 1 (A) → [0, 0] (reference)
# Level 2 (B) → [1, 0]
# Level 3 (C) → [0, 1]

println("\ngroup3 contrast mapping (DummyCoding drops first):")
println("  Level 1 (A) → [0, 0] (reference)")
println("  Level 2 (B) → [1, 0]")
println("  Level 3 (C) → [0, 1]")

println("\ngroup4 contrast mapping (DummyCoding drops first):")
println("  Level 1 (W) → [0, 0, 0] (reference)")
println("  Level 2 (X) → [1, 0, 0]")
println("  Level 3 (Y) → [0, 1, 0]")
println("  Level 4 (Z) → [0, 0, 1]")

# So for row 250 with B and Y (levels 2 and 3):
# group3 → [1, 0]
# group4 → [0, 1, 0]
# Kronecker product should give:
# Position [1,2] in the 2×3 grid

println("\nFor row $row:")
println("  group3 = B (level $g3_level) → contrast [1, 0]")
println("  group4 = Y (level $g4_level) → contrast [0, 1, 0]")
println("\nKronecker product positions that should be 1:")
println("  group3[1] × group4[2] = 1 × 1 = 1")
println("  This maps to position 2 in the flattened order")
println("  Which is 'group3: B & group4: Y'")

# Now let's check what positions we're actually computing
println("\n" * "="^60)
println("Checking actual computation:")
println("="^60)

# The Kronecker expansion we do is:
# [1,1], [1,2], [1,3], [2,1], [2,2], [2,3]
# Which maps to:
# B&X,   B&Y,   B&Z,   C&X,   C&Y,   C&Z

positions = [[1,2], [1,2,3]]
combos = FormulaCompiler.compute_all_interaction_combinations(positions)

println("\nOur position combinations:")
for (i, combo) in enumerate(combos)
    g3_idx = combo[1]
    g4_idx = combo[2]
    
    g3_name = g3_idx == 1 ? "B" : "C"
    g4_name = g4_idx == 1 ? "X" : (g4_idx == 2 ? "Y" : "Z")
    
    println("  Combo $i: positions $combo → group3:$g3_name & group4:$g4_name")
end

# The issue seems to be that we're using position indices directly
# But we need to check which contrast column is active based on the level