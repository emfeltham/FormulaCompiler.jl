# Debug categorical level ordering
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays
using FormulaCompiler: make_test_data
using Random

Random.seed!(08540)

# Use the same test data
n = 500
df = make_test_data(; n)

println("Categorical Variable Levels:")
println("="^50)

println("\ngroup3 levels: ", levels(df.group3))
println("group4 levels: ", levels(df.group4))
println("group5 levels: ", levels(df.group5))

# Check row 250 specifically
println("\nRow 250 data:")
println("  group3 = $(df.group3[250]) (level code: $(levelcode(df.group3[250])))")
println("  group4 = $(df.group4[250]) (level code: $(levelcode(df.group4[250])))")

# Let's see what the contrast matrices look like
using StatsModels
formula = @formula(y ~ group3 * group4)
model = lm(formula, df)
sch = StatsModels.schema(formula, df)

println("\nContrast matrices from schema:")
println("group3 contrast:")
display(sch[Term(:group3)].contrasts.matrix)
println("\ngroup4 contrast:")
display(sch[Term(:group4)].contrasts.matrix)

# Check the interaction term structure
println("\nInteraction term columns:")
for (i, name) in enumerate(coefnames(model))
    if contains(name, "&")
        println("  $i: $name")
    end
end

# Now let's manually compute what row 250 should produce
println("\n" * "="^50)
println("Manual computation for row 250:")
g3_val = df.group3[250]
g4_val = df.group4[250]
println("group3 = $g3_val, group4 = $g4_val")

# group3: B means contrast [1, 0] (B=1, C=0)
# group4: Y means contrast [0, 1, 0] (X=0, Y=1, Z=0)
println("\nExpected contrast values:")
println("group3: B → [1, 0]")
println("group4: Y → [0, 1, 0]")

println("\nExpected interaction (Kronecker product):")
println("group3: B & group4: X → 1 * 0 = 0")
println("group3: C & group4: X → 0 * 0 = 0")
println("group3: B & group4: Y → 1 * 1 = 1")  # This should be 1!
println("group3: C & group4: Y → 0 * 1 = 0")
println("group3: B & group4: Z → 1 * 0 = 0")
println("group3: C & group4: Z → 0 * 0 = 0")