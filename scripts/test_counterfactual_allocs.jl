using FormulaCompiler, GLM, DataFrames, Tables, BenchmarkTools, ForwardDiff

# Simple model
df = DataFrame(y = randn(100), x = randn(100), z = randn(100))
model = lm(@formula(y ~ x + z), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build AD evaluator
vars = [:x, :z]
de = FormulaCompiler.derivativeevaluator_ad(compiled, data, vars)

# Setup dual numbers
x_dual_vec = de.x_dual_vec
partials_unit_vec = de.partials_unit_vec
for i in 1:2
    base_val = Float64(getproperty(de.base_data, de.vars[i])[1])
    x_dual_vec[i] = typeof(x_dual_vec[i])(base_val, partials_unit_vec[i])
end

println("=== Testing update_counterfactual_for_var! ===")
b = @benchmark FormulaCompiler.update_counterfactual_for_var!(
    $(de.counterfactuals),
    $(de.vars),
    $(de.vars[1]),
    1,
    $(x_dual_vec[1])
)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

println("\n=== Testing loop with both updates ===")
b = @benchmark begin
    for i in 1:2
        FormulaCompiler.update_counterfactual_for_var!(
            $(de.counterfactuals),
            $(de.vars),
            $(de.vars)[i],
            1,
            $(x_dual_vec)[i]
        )
    end
end
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

# Test compiled_dual evaluation
println("\n=== Testing compiled_dual evaluation ===")
rowvec_dual_vec = de.rowvec_dual_vec
b = @benchmark $(de.compiled_dual)($rowvec_dual_vec, $(de.data_counterfactual), 1)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

# Test full derivative_modelrow!
J = Matrix{Float64}(undef, length(compiled), 2)
println("\n=== Testing full derivative_modelrow! ===")
b = @benchmark FormulaCompiler.derivative_modelrow!($J, $de, 1)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")
