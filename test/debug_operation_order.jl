# Debug the actual operations generated for interactions
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays
using FormulaCompiler: decompose_formula, compile_formula
using Random

Random.seed!(08540)

# Small test case
df = DataFrame(
    x = [1.0, 2.0],
    group3 = categorical(["A", "B"]),
    group4 = categorical(["W", "X"]),
    resp = randn(2)
)
data = Tables.columntable(df)

# Simple categorical interaction
formula = @formula(resp ~ group3 * group4)
model = lm(formula, df)

println("="^70)
println("DEBUGGING OPERATION ORDER FOR CATEGORICAL INTERACTIONS")
println("="^70)

println("\nData:")
display(df)

println("\nModel matrix:")
mm = modelmatrix(model)
display(mm)

println("\nColumn names:")
for (i, name) in enumerate(coefnames(model))
    println("  $i: $name")
end

# Compile and examine operations
compiled = compile_formula(model, data)

println("\n" * "="^70)
println("Generated Operations")
println("="^70)

# Get the operations tuple
ops = compiled.ops

println("\nTotal operations: $(length(ops))")
println("\nOperation details:")

for (i, op) in enumerate(ops)
    op_type = typeof(op).name.name
    
    if op_type == :ContrastOp
        col = typeof(op).parameters[1]
        positions = typeof(op).parameters[2]
        println("  $i. ContrastOp{:$col, $positions}")
        println("     Produces contrasts for :$col at positions $positions")
    elseif op_type == :BinaryOp
        func = typeof(op).parameters[1]
        in1 = typeof(op).parameters[2]
        in2 = typeof(op).parameters[3]
        out = typeof(op).parameters[4]
        println("  $i. BinaryOp{:$func, $in1, $in2, $out}")
        println("     scratch[$out] = scratch[$in1] $func scratch[$in2]")
    elseif op_type == :CopyOp
        from = typeof(op).parameters[1]
        to = typeof(op).parameters[2]
        println("  $i. CopyOp{$from, $to}")
        println("     output[$to] = scratch[$from]")
    elseif op_type == :ConstantOp
        val = typeof(op).parameters[1]
        pos = typeof(op).parameters[2]
        println("  $i. ConstantOp{$val, $pos}")
        println("     scratch[$pos] = $val")
    else
        println("  $i. $op_type: $op")
    end
end

# Now trace execution for row 2 (B, X)
println("\n" * "="^70)
println("Execution Trace for Row 2 (group3=B, group4=X)")
println("="^70)

# Manual trace
scratch_size = typeof(compiled).parameters[2]  # Extract from type parameter
scratch = zeros(Float64, scratch_size)
output = zeros(Float64, length(compiled))

println("\nRow 2 data:")
println("  group3 = $(df.group3[2]) (level $(levelcode(df.group3[2])))")
println("  group4 = $(df.group4[2]) (level $(levelcode(df.group4[2])))")

# Execute and show scratch state
compiled(output, data, 2)

println("\nFinal output:")
for (i, val) in enumerate(output)
    println("  output[$i] ($(coefnames(model)[i])): $val")
end

println("\nExpected from modelmatrix:")
for (i, val) in enumerate(mm[2, :])
    println("  mm[2,$i] ($(coefnames(model)[i])): $val")
end

# Check if they match
if !isapprox(output, mm[2, :], rtol=1e-10)
    println("\n❌ MISMATCH DETECTED!")
    diff = output .- mm[2, :]
    for (i, d) in enumerate(diff)
        if abs(d) > 1e-10
            println("  Position $i: diff = $d")
        end
    end
end