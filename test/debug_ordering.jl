# Debug ordering issue in categorical interactions
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays
using StatsModels

println("="^70)
println("DEBUGGING INTERACTION ORDERING")
println("="^70)

# Simple test case: x * group3 where group3 has A, B, C levels
n = 6
df = DataFrame(
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    group3 = categorical(["A", "B", "C", "A", "B", "C"]),
    y = randn(n)
)
data = Tables.columntable(df)

# Test simple interaction first
formula = @formula(y ~ x * group3)
model = lm(formula, df)
mm = modelmatrix(model)

println("Formula: y ~ x * group3")
println("Model matrix columns: ", coefnames(model))
println("Model matrix for row 2 (x=2, group3=B):")
println(mm[2, :])

# Compile and test
compiled = compile_formula(model, data)
output = Vector{Float64}(undef, length(compiled))

compiled(output, data, 2)
println("\nCompiled output for row 2:")
println(output)

println("\nDifference:")
diff = output .- mm[2, :]
for (i, d) in enumerate(diff)
    if abs(d) > 1e-10
        println("Position $i: diff = $d")
        println("  Expected: $(mm[2, i]) ($(coefnames(model)[i]))")
        println("  Got: $(output[i])")
    end
end

# Now let's test the Kronecker pattern directly
println("\n" * "="^70)
println("TESTING KRONECKER PATTERN")
println("="^70)

# Our compute_interaction_pattern function
pattern = FormulaCompiler.compute_interaction_pattern(1, 2)  # x (scalar) * group3 (2 contrasts)
println("Our pattern for 1×2: ", pattern)

# StatsModels uses kron(b, a) where a varies fast, b varies slow
# For x * group3:
# - x is scalar (width 1)
# - group3 has 2 contrasts (B and C, dropping A)
# Result should be: [x*B, x*C]

# But the pattern tells us how to combine indices
# Pattern [(1,1), (1,2)] means:
# - First output: component1[1] * component2[1] = x * group3_B
# - Second output: component1[1] * component2[2] = x * group3_C

println("\nExpected interaction columns:")
println("1. x:group3[B] (when group3=B, value should be x)")
println("2. x:group3[C] (when group3=C, value should be x)")

# Test with a more complex case
println("\n" * "="^70)
println("TESTING group3 * group4")
println("="^70)

df2 = DataFrame(
    group3 = categorical(["A", "B", "C", "A", "B", "C"]),
    group4 = categorical(["P", "Q", "R", "S", "P", "Q"]),
    y = randn(6)
)
data2 = Tables.columntable(df2)

formula2 = @formula(y ~ group3 * group4)
model2 = lm(formula2, df2)
mm2 = modelmatrix(model2)

println("Formula: y ~ group3 * group4")
println("Model matrix columns: ", coefnames(model2))

# The interaction columns should be:
# group3[B] * group4[Q,R,S] = 3 columns
# group3[C] * group4[Q,R,S] = 3 columns
# Total = 6 interaction columns

compiled2 = compile_formula(model2, data2)
output2 = Vector{Float64}(undef, length(compiled2))

# Test row 2: group3=B, group4=Q
compiled2(output2, data2, 2)
println("\nRow 2 (group3=B, group4=Q):")
println("Expected: ", mm2[2, :])
println("Got:      ", output2)

# Check which positions differ
diff2 = output2 .- mm2[2, :]
problem_positions = findall(abs.(diff2) .> 1e-10)
if !isempty(problem_positions)
    println("\nMismatched positions: ", problem_positions)
    for pos in problem_positions
        println("Position $pos ($(coefnames(model2)[pos])):")
        println("  Expected: $(mm2[2, pos])")
        println("  Got: $(output2[pos])")
    end
end