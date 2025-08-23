# derivative_generate.jl
# modelrow style updates for the derivative

using FormulaCompiler
using GLM, DataFrames, Tables

# Create test data
df = DataFrame(x = randn(100), y = randn(100), z = abs.(randn(100)) .+ 0.1);

# Test derivative compilation
model = lm(@formula(y ~ x + log(z)), df)
compiled = compile_formula(model)
derivative_x = compile_derivative_formula(compiled, :x)

# Test evaluation
data = Tables.columntable(df);
row_vec = Vector{Float64}(undef, length(derivative_x))
derivative_x(row_vec, data, 1)

println("✅ Derivative compilation working!")
println("∂/∂x = ", row_vec)