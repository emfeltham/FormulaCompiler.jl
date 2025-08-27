# More detailed debug for derivative issues
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

# Build derivative evaluator
vars = [:x, :z]
de = build_derivative_evaluator(compiled, data; vars=vars)

# Test the DerivClosure directly
println("Testing DerivClosure directly:")

# Create the closure
g = de.g

println("de object id: ", objectid(de))
println("g.de_ref[] object id: ", objectid(g.de_ref[]))
println("Are they the same? ", de === g.de_ref[])

# Test at row 3 (group=C)
de.row = 3
println("After setting de.row = 3:")
println("de.row = ", de.row)
println("g.de_ref[].row = ", g.de_ref[].row)
x_vec = [data.x[3], data.z[3]]
println("\nRow 3 (group=$(data.group3[3])):")
println("Input x: ", x_vec)

# Test with Float64
result_float = g(x_vec)
println("Float64 result: ", result_float)

# Test with Dual
tag = ForwardDiff.Tag{:test, Float64}
DualType = ForwardDiff.Dual{tag, Float64, 2}
x_dual = [
    DualType(x_vec[1], ForwardDiff.Partials((1.0, 0.0))),  # dx/dx = 1, dx/dz = 0
    DualType(x_vec[2], ForwardDiff.Partials((0.0, 1.0))),  # dz/dx = 0, dz/dz = 1
]
result_dual = g(x_dual)
println("Dual result values: ", [ForwardDiff.value(r) for r in result_dual])
println("Dual result partials (w.r.t. x): ", [ForwardDiff.partials(r)[1] for r in result_dual])

# Now test at row 5 (group=A)
de.row = 5
x_vec = [data.x[5], data.z[5]]
println("\nRow 5 (group=$(data.group3[5])):")
println("Input x: ", x_vec)

# Test with Float64
result_float = g(x_vec)
println("Float64 result: ", result_float)

# Test with Dual
x_dual = [
    DualType(x_vec[1], ForwardDiff.Partials((1.0, 0.0))),
    DualType(x_vec[2], ForwardDiff.Partials((0.0, 1.0))),
]
result_dual = g(x_dual)
println("Dual result values: ", [ForwardDiff.value(r) for r in result_dual])
println("Dual result partials (w.r.t. x): ", [ForwardDiff.partials(r)[1] for r in result_dual])

# Check if the issue is in the override system
println("\n\nChecking override system:")

# Force a Dual evaluation to populate the cache
de.row = 3
x_dual_test = [
    DualType(data.x[3], ForwardDiff.Partials((1.0, 0.0))),
    DualType(data.z[3], ForwardDiff.Partials((0.0, 1.0))),
]
_ = g(x_dual_test)
println("After first call:")
println("de.overrides_dual type: ", typeof(de.overrides_dual))
if de.overrides_dual !== nothing
    println("First override row: ", de.overrides_dual[1].row)
end

# Now change row and call again
de.row = 5
x_dual_test2 = [
    DualType(data.x[5], ForwardDiff.Partials((1.0, 0.0))),
    DualType(data.z[5], ForwardDiff.Partials((0.0, 1.0))),
]
_ = g(x_dual_test2)
println("\nAfter second call (different row):")
if de.overrides_dual !== nothing
    println("First override row: ", de.overrides_dual[1].row)
end