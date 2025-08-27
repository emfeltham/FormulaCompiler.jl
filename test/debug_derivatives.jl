# Debug script for derivative issues
using FormulaCompiler
using DataFrames, Tables, GLM, CategoricalArrays
using Random
using ForwardDiff

Random.seed!(12345)

# Create test data
n = 10
df = DataFrame(
    y = randn(n),
    x = randn(n),
    z = abs.(randn(n)) .+ 0.1,
    group3 = categorical(["A", "B", "C", "A", "A", "B", "C", "A", "B", "C"]),
)
data = Tables.columntable(df)
model = lm(@formula(y ~ 1 + x + z + x & group3), df)
compiled = compile_formula(model, data)

println("Data preview:")
println("Row | group3 | x")
for i in 1:10
    println("$i   | $(data.group3[i]) | $(round(data.x[i], digits=3))")
end

# Build derivative evaluator
vars = [:x, :z]
de = build_derivative_evaluator(compiled, data; vars=vars)

# Test specific rows
test_rows = [3, 5]
for row in test_rows
    println("\n=== Row $row (group=$(data.group3[row])) ===")
    
    # AD Jacobian
    J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
    derivative_modelrow!(J_ad, de, row)
    println("AD Jacobian, col 1: ", J_ad[:, 1])
    
    # FD Jacobian
    J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
    derivative_modelrow_fd!(J_fd, de, row)
    println("FD Jacobian, col 1: ", J_fd[:, 1])
    
    # Direct test with Dual numbers
    println("\nDirect Dual test:")
    x_val = data.x[row]
    z_val = data.z[row]
    
    # Create dual for x (differentiate w.r.t. x only)
    tag = ForwardDiff.Tag{:test, Float64}
    DualType = ForwardDiff.Dual{tag, Float64, 1}
    x_dual = DualType(x_val, ForwardDiff.Partials((1.0,)))  # partial = 1 for x
    z_dual = DualType(z_val, ForwardDiff.Partials((0.0,)))  # partial = 0 for z
    
    # Create override data with Dual types
    x_vec = [i == row ? x_dual : DualType(data.x[i], ForwardDiff.Partials((0.0,))) for i in 1:n]
    z_vec = [i == row ? z_dual : DualType(data.z[i], ForwardDiff.Partials((0.0,))) for i in 1:n]
    
    data_dual = merge(data, (x=x_vec, z=z_vec))
    
    # Need to create a Dual-typed compiled instance
    Ops = typeof(compiled).parameters[2]
    S = typeof(compiled).parameters[3]
    O = typeof(compiled).parameters[4]
    compiled_dual = UnifiedCompiled{DualType, Ops, S, O}(compiled.ops)
    
    # Evaluate with Dual
    output_dual = Vector{DualType}(undef, length(compiled))
    compiled_dual(output_dual, data_dual, row)
    
    # Extract derivatives
    derivs = [ForwardDiff.partials(output_dual[i])[1] for i in 1:length(output_dual)]
    println("Direct Dual derivs: ", derivs)
    
    # Check the terms
    println("\nTerm values at row $row:")
    output_float = Vector{Float64}(undef, length(compiled))
    compiled(output_float, data, row)
    println("Terms: ", output_float)
    println("Term 4 is x*group3:B, should be $(data.x[row]) * $(data.group3[row] == "B" ? 1 : 0) = $(data.x[row] * (data.group3[row] == "B" ? 1 : 0))")
end