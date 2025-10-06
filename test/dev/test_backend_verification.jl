# Verify which backend is actually being used
using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays

# Simple test data
n = 100
df = DataFrame(
    y = rand(Bool, n),
    x1 = randn(n),
    x2 = randn(n)
)

# Fit logistic model
model = glm(@formula(y ~ x1 + x2), df, Binomial(), LogitLink())

# Compile
data = Tables.columntable(df)
compiled = compile_formula(model, data)
β = coef(model)

# Build evaluators for both backends
vars = [:x1, :x2]
de_ad = derivativeevaluator(:ad, compiled, data, vars)
de_fd = derivativeevaluator(:fd, compiled, data, vars)

# Test with single row
rows = [1]
var = :x1
gβ_sum = zeros(length(de_ad))

# Test with concrete type dispatch
@debug "=== Testing concrete type dispatch ==="

# Test FD evaluator
@debug "\nUsing FD evaluator: ", typeof(de_fd)
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de_fd, β, rows, var; link=LogitLink(), backend=:fd)

# Test AD evaluator
@debug "\nUsing AD evaluator: ", typeof(de_ad)
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de_ad, β, rows, var; link=LogitLink(), backend=:ad)

# Now let's check if the results are different
gβ_fd = zeros(length(de_fd))
gβ_ad = zeros(length(de_ad))

FormulaCompiler.accumulate_ame_gradient!(gβ_fd, de_fd, β, rows, var; link=LogitLink(), backend=:fd)
FormulaCompiler.accumulate_ame_gradient!(gβ_ad, de_ad, β, rows, var; link=LogitLink(), backend=:ad)

@debug "\n=== Results comparison ==="
@debug "FD result: ", gβ_fd[1:3]
@debug "AD result: ", gβ_ad[1:3]
@debug "Are they identical? ", gβ_fd ≈ gβ_ad
@debug "Max difference: ", maximum(abs.(gβ_fd - gβ_ad))

# Test with more rows to see performance difference
rows = 1:50
@debug "\n=== Performance with 50 rows ==="

# Count allocations using concrete evaluators
function count_allocs_fd()
    gβ = zeros(length(de_fd))
    FormulaCompiler.accumulate_ame_gradient!(gβ, de_fd, β, rows, var; link=LogitLink(), backend=:fd)
end

function count_allocs_ad()
    gβ = zeros(length(de_ad))
    FormulaCompiler.accumulate_ame_gradient!(gβ, de_ad, β, rows, var; link=LogitLink(), backend=:ad)
end

# Warm up
count_allocs_fd()
count_allocs_ad()

# Measure
@debug "FD allocations:"
@time count_allocs_fd()

@debug "AD allocations:"
@time count_allocs_ad()