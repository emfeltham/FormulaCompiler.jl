# Compare AD vs FD performance for single column
using FormulaCompiler, GLM, DataFrames, Tables, BenchmarkTools, CategoricalArrays

# Create test data
n = 10000
df = DataFrame(
    y = rand(Bool, n),
    x1 = randn(n),
    x2 = randn(n),
    x3 = randn(n),
    group = categorical(rand(["A", "B", "C"], n))
)

# Fit logistic regression model
model = glm(@formula(y ~ x1 * x2 * x3 * group), df, Binomial(), LogitLink())
@debug "Model has $(length(coef(model))) parameters"

# Compile formula
data = Tables.columntable(df)
compiled = compile_formula(model, data)
β = coef(model)

# Build derivative evaluators
vars = [:x1, :x2, :x3]
de_fd = derivativeevaluator(:fd, compiled, data, vars)
de_ad = derivativeevaluator(:ad, compiled, data, vars)

# Test single row gradient computation
rows = 1:100
var = :x1

# Allocate gradient buffer
gβ_sum_fd = zeros(length(de_fd))
gβ_sum_ad = zeros(length(de_ad))

@debug "\n=== Performance comparison for $(length(rows)) rows ==="

# Warm up
FormulaCompiler.accumulate_ame_gradient!(gβ_sum_fd, de_fd, β, rows, var; link=LogitLink(), backend=:fd)
FormulaCompiler.accumulate_ame_gradient!(gβ_sum_ad, de_ad, β, rows, var; link=LogitLink(), backend=:ad)

# FD benchmark
@debug "FD backend (2 formula evaluations per row):"
@btime FormulaCompiler.accumulate_ame_gradient!(
    $gβ_sum_fd, $de_fd, $β, $rows, $var;
    link=$LogitLink(), backend=:fd
)

# AD benchmark
@debug "AD backend (1 full Jacobian per row):"
@btime FormulaCompiler.accumulate_ame_gradient!(
    $gβ_sum_ad, $de_ad, $β, $rows, $var;
    link=$LogitLink(), backend=:ad
)

# Check accuracy
@debug "\nMax difference between FD and AD: ", maximum(abs.(gβ_sum_fd - gβ_sum_ad))

# Now let's see what happens with more variables
@debug "\n=== With $(length(vars)) variables to differentiate ==="

# The AD computes full Jacobian (all 3 columns) even when we only need 1
# The FD computes only the column we need

@debug "FD extracts 1 column (efficient for single var)"
@debug "AD computes all $(length(vars)) columns (overhead for single var)"

# Let's test when we need all variables
@debug "\n=== If we needed all $(length(vars)) variables ==="

all_grads_fd = zeros(length(de_fd), length(vars))
all_grads_ad = zeros(length(de_ad), length(vars))

# FD: Need to loop over variables
print("FD ($(length(vars)) separate calls): ")
@time for (i, v) in enumerate(vars)
    FormulaCompiler.accumulate_ame_gradient!(
        view(all_grads_fd, :, i), de_fd, β, rows, v;
        link=LogitLink(), backend=:fd
    )
end

# AD: Can reuse the Jacobian
print("AD (could be optimized to reuse Jacobian): ")
@time for (i, v) in enumerate(vars)
    FormulaCompiler.accumulate_ame_gradient!(
        view(all_grads_ad, :, i), de_ad, β, rows, v;
        link=LogitLink(), backend=:ad
    )
end

@debug "\nKey insight: AD computes full Jacobian even for 1 variable"
@debug "This is overhead when we only need 1 column out of $(length(vars))"
@debug "For models with many parameters, full Jacobian is expensive"
