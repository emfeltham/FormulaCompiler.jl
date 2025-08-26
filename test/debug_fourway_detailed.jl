# Detailed debug of four-way interaction issue
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays

println("="^70)
println("DETAILED FOUR-WAY INTERACTION DEBUG")
println("="^70)

# Minimal case that shows the problem
n = 6
df = DataFrame(
    x = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    y = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    group3 = categorical(["A", "B", "C", "A", "B", "C"]),
    group4 = categorical(["P", "Q", "P", "Q", "P", "Q"]),
    resp = randn(n)
)
data = Tables.columntable(df)

# Four-way interaction formula
formula = @formula(resp ~ x * y * group3 * group4)
model = lm(formula, df)
mm = modelmatrix(model)

println("Data:")
for i in 1:n
    println("Row $i: x=$(df.x[i]), y=$(df.y[i]), group3=$(df.group3[i]), group4=$(df.group4[i])")
end

println("\nModel matrix size: ", size(mm))
println("Column names: ")
for (i, name) in enumerate(coefnames(model))
    println("  $i: $name")
end

# Compile formula
compiled = compile_formula(model, data)
output = Vector{Float64}(undef, length(compiled))

# Find the problematic row from earlier test
println("\n" * "="^70)
println("Testing each row:")
println("="^70)

for row in 1:n
    compiled(output, data, row)
    expected = mm[row, :]
    
    diff = output .- expected
    problem_positions = findall(abs.(diff) .> 1e-10)
    
    if isempty(problem_positions)
        println("Row $row: ✅ Correct")
    else
        println("Row $row: ❌ Mismatched")
        println("  Data: x=$(df.x[row]), y=$(df.y[row]), group3=$(df.group3[row]), group4=$(df.group4[row])")
        
        # Group mismatches by interaction order
        for pos in problem_positions
            col_name = coefnames(model)[pos]
            println("  Position $pos ($col_name):")
            println("    Expected: $(expected[pos])")
            println("    Got: $(output[pos])")
            println("    Diff: $(diff[pos])")
        end
    end
end

# Let's specifically look at the 4-way interaction terms
println("\n" * "="^70)
println("Analyzing 4-way interaction terms only")
println("="^70)

# Find indices of 4-way interaction terms (they contain 3 '&' symbols)
fourway_indices = Int[]
for (i, name) in enumerate(coefnames(model))
    if count(==('&'), name) == 3
        push!(fourway_indices, i)
        println("4-way term at position $i: $name")
    end
end

# Check if the issue is specific to certain categorical combinations
println("\nChecking pattern in mismatches...")

# Row 3 from earlier had issues - let's see what makes it special
if n >= 3
    println("\nRow 3 analysis:")
    println("Data: x=$(df.x[3]), y=$(df.y[3]), group3=$(df.group3[3]), group4=$(df.group4[3])")
    
    compiled(output, data, 3)
    expected = mm[3, :]
    
    # Focus on 4-way terms
    println("4-way interaction values:")
    for idx in fourway_indices
        exp_val = expected[idx]
        got_val = output[idx]
        if !isapprox(exp_val, got_val, rtol=1e-10)
            println("  ❌ $(coefnames(model)[idx]): expected=$exp_val, got=$got_val")
        else
            println("  ✅ $(coefnames(model)[idx]): $exp_val")
        end
    end
end

# Check the pattern: are we getting the wrong categorical level combinations?
println("\n" * "="^70)
println("Hypothesis: Kronecker product ordering issue in 4-way")
println("="^70)

# In our Kronecker expansion for A×B×C×D, we do:
# 1. First compute A×B → intermediate1
# 2. Then compute intermediate1×C → intermediate2  
# 3. Finally compute intermediate2×D → final

# But the ordering might not match StatsModels' expectation
println("Our cascading approach:")
println("1. x * y → scalar")
println("2. (x*y) * group3 → 2 values (for B,C levels)")
println("3. ((x*y)*group3) * group4 → 2 values (for Q level)")
println("")
println("Expected by StatsModels might be different ordering...")

# Test with simpler 3-way to understand the pattern
formula3 = @formula(resp ~ x * y * group3)
model3 = lm(formula3, df)
mm3 = modelmatrix(model3)

compiled3 = compile_formula(model3, data)
output3 = Vector{Float64}(undef, length(compiled3))

println("\n3-way interaction test (x * y * group3):")
for row in 1:min(3, n)
    compiled3(output3, data, row)
    expected3 = mm3[row, :]
    
    diff3 = output3 .- expected3
    if any(abs.(diff3) .> 1e-10)
        println("Row $row: Has differences")
        problem_pos = findall(abs.(diff3) .> 1e-10)
        for pos in problem_pos
            println("  Position $pos ($(coefnames(model3)[pos])): diff = $(diff3[pos])")
        end
    else
        println("Row $row: ✅")
    end
end