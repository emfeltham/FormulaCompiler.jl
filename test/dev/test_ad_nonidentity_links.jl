# Test AD support for non-identity link functions
using FormulaCompiler, GLM, DataFrames, Tables, Test, BenchmarkTools, CategoricalArrays

# Create test data
n = 1000
df = DataFrame(
    y = rand(Bool, n),
    x1 = randn(n),
    x2 = randn(n),
    group = categorical(rand(["A", "B", "C"], n))
)

# Fit logistic regression model
model = glm(@formula(y ~ x1 + x2 + group), df, Binomial(), LogitLink())

# Compile formula
data = Tables.columntable(df)
compiled = compile_formula(model, data)
β = coef(model)

# Build derivative evaluators
vars = [:x1, :x2]
de_fd = derivativeevaluator(:fd, compiled, data, vars)
de_ad = derivativeevaluator(:ad, compiled, data, vars)

# Test single row gradient computation with both backends
row = 1
var = :x1

# Allocate gradient buffer
gβ_sum_fd = zeros(length(de_fd))
gβ_sum_ad = zeros(length(de_ad))

# Test FD backend
@time FormulaCompiler.accumulate_ame_gradient!(
    gβ_sum_fd, de_fd, β, [row], var;
    link=LogitLink(), backend=:fd
)

# Test AD backend (should now work!)
@time FormulaCompiler.accumulate_ame_gradient!(
    gβ_sum_ad, de_ad, β, [row], var;
    link=LogitLink(), backend=:ad
)

# Check they give similar results
@test isapprox(gβ_sum_fd, gβ_sum_ad; rtol=1e-6)

@debug "FD gradient: ", gβ_sum_fd[1:5]
@debug "AD gradient: ", gβ_sum_ad[1:5]

# Benchmark with larger row set
rows = 1:100

@debug "\nBenchmarking with 100 rows:"

# FD benchmark
@debug "FD backend:"
@btime FormulaCompiler.accumulate_ame_gradient!(
    $gβ_sum_fd, $de_fd, $β, $rows, $var;
    link=$LogitLink(), backend=:fd
)

# AD benchmark
@debug "AD backend:"
@btime FormulaCompiler.accumulate_ame_gradient!(
    $gβ_sum_ad, $de_ad, $β, $rows, $var;
    link=$LogitLink(), backend=:ad
)
