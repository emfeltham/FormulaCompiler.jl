# Test AD allocations
using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays, BenchmarkTools

# Simple model
n = 1000
df = DataFrame(
    y = rand(Bool, n),
    x1 = randn(n),
    x2 = randn(n),
    group = categorical(rand(["A", "B"], n))
)

model = glm(@formula(y ~ x1 + x2 + group), df, Binomial(), LogitLink())
@debug "Model with $(length(coef(model))) parameters"

# Compile
data = Tables.columntable(df)
compiled = compile_formula(model, data)
β = coef(model)

# Build evaluators
vars = [:x1, :x2]
de_ad = derivativeevaluator(:ad, compiled, data, vars)
de_fd = derivativeevaluator(:fd, compiled, data, vars)

# Test allocations for derivative_modelrow!
J = Matrix{Float64}(undef, length(compiled), length(vars))
row = 1

# Warm up
FormulaCompiler.derivative_modelrow!(J, de_ad, row)
FormulaCompiler.derivative_modelrow!(J, de_ad, row)

@debug "\n=== Testing derivative_modelrow! allocations ==="
@debug "Should be zero-allocation after warmup..."

# Test single call
print("Single call: ")
@time FormulaCompiler.derivative_modelrow!(J, de_ad, row)

# Benchmark
@debug "Benchmark:"
@btime FormulaCompiler.derivative_modelrow!($J, $de_ad, $row)

# Now test our chain rule implementation in accumulate_ame_gradient!
gβ = zeros(length(de_ad))

@debug "\n=== Testing accumulate_ame_gradient! with AD ==="

# Single row
rows = [1]
var = :x1

# Warm up
FormulaCompiler.accumulate_ame_gradient!(gβ, de_ad, β, rows, var; link=LogitLink(), backend=:ad)

print("Single row with AD: ")
@time FormulaCompiler.accumulate_ame_gradient!(gβ, de_ad, β, rows, var; link=LogitLink(), backend=:ad)

# Benchmark
@btime FormulaCompiler.accumulate_ame_gradient!($gβ, $de_ad, $β, $rows, $var; link=$(LogitLink()), backend=:ad)

# Test with more rows
rows = 1:100
@debug "\n100 rows with AD:"
@btime FormulaCompiler.accumulate_ame_gradient!($gβ, $de_ad, $β, $rows, $var; link=$(LogitLink()), backend=:ad)

# Compare to FD
@debug "\n100 rows with FD:"
@btime FormulaCompiler.accumulate_ame_gradient!($gβ, $de_fd, $β, $rows, $var; link=$(LogitLink()), backend=:fd)