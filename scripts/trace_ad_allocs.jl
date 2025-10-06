using FormulaCompiler, GLM, DataFrames, Tables, BenchmarkTools

# Simple model to trace allocations
df = DataFrame(y = randn(100), x = randn(100), z = randn(100))
model = lm(@formula(y ~ x + z), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build AD evaluator (revert to ForwardDiff.Dual)
vars = [:x, :z]

println("Building AD evaluator...")
de = FormulaCompiler.derivativeevaluator_ad(compiled, data, vars)

J = Matrix{Float64}(undef, length(compiled), length(vars))

# Warmup
println("Warmup...")
for i in 1:100
    FormulaCompiler.derivative_modelrow!(J, de, 1)
end

println("\nTesting derivative_modelrow! allocations...")
b = @benchmark FormulaCompiler.derivative_modelrow!($J, $de, 1)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")
println("Time: $(minimum(b).time) ns")

# Now test each component separately
println("\n=== Component analysis ===")

println("\n1. Testing x_dual_vec construction:")
x_dual_vec = de.x_dual_vec
partials_unit_vec = de.partials_unit_vec
base_data = de.base_data
vars_list = de.vars

b = @benchmark begin
    for i in 1:2
        base_val = getproperty($base_data, $vars_list[i])[1]
        $x_dual_vec[i] = typeof($x_dual_vec[i])(Float64(base_val), $partials_unit_vec[i])
    end
end
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

println("\n2. Testing counterfactual update:")
b = @benchmark begin
    for i in 1:2
        FormulaCompiler.update_counterfactual_for_var!(
            $(de.counterfactuals),
            $(de.vars),
            $(de.vars[i]),
            1,
            $(de.x_dual_vec[i])
        )
    end
end
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

println("\n3. Testing compiled_dual evaluation:")
rowvec_dual_vec = de.rowvec_dual_vec
compiled_dual = de.compiled_dual
data_cf = de.data_counterfactual

b = @benchmark $compiled_dual($rowvec_dual_vec, $data_cf, 1)
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")

println("\n4. Testing gradient extraction:")
b = @benchmark begin
    for i in 1:size($J, 1)
        parts = ForwardDiff.partials($rowvec_dual_vec[i])
        for j in 1:2
            $J[i,j] = parts[j]
        end
    end
end
println("Memory: $(minimum(b).memory) bytes")
println("Allocs: $(minimum(b).allocs)")
