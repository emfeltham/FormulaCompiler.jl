# override_usage.jl

using Revise
using BenchmarkTools
using EfficientModelMatrices

using DataFrames, GLM, Tables, CategoricalArrays
using CategoricalArrays: CategoricalValue, levelcode
using StatsModels
using StandardizedPredictors: ZScoredTerm
using Random
using StandardizedPredictors

Random.seed!(06515)

# 1. Create your data and model
df = DataFrame(
    x = randn(1000),
    y = randn(1000),
    z = abs.(randn(1000)) .+ 0.1,
    group = categorical(rand(["A", "B", "C"], 1000))
);

contrasts = Dict(:x => ZScore());

df.bool = rand([false, true], nrow(df));
df.group2 = categorical(rand(["C", "D", "X"], nrow(df)));
df.group3 = categorical(rand(["E", "F", "G"], nrow(df)))
df.cat2a = categorical(rand(["X", "Y"], nrow(df)));
df.cat2b = categorical(rand(["P", "Q"], nrow(df)));
data = Tables.columntable(df);

model = lm(@formula(y ~ x + x^2 * log(z) + group), df);

## base scenario
row_idx = 1
compiled = compile_formula(model);
row_vec = Vector{Float64}(undef, length(compiled));
compiled(row_vec, data, row_idx);
rv1 = deepcopy(row_vec);
fill!(row_vec, 0.0);
modelrow!(row_vec, compiled, data, row_idx);
mm = modelmatrix(model);
@assert row_vec == vec(mm[row_idx, :])

@assert rv1 == row_vec # check that they are the same

# Override single variable
scenario1 = create_scenario("x_at_mean", data, Dict(:x => mean(data.x)));
# Use with compiled formula
compiled = compile_formula(model);
row_vec = Vector{Float64}(undef, length(compiled));
@btime compiled(row_vec, scenario1.data, 1);

# Override multiple variables  
scenario2 = create_scenario("policy", data, Dict(:x => 2.5, :group => "A"));
# Use with compiled formula
compiled = compile_formula(model);
row_vec = Vector{Float64}(undef, length(compiled));
@btime compiled(row_vec, scenario2.data, 1);

levels(data.group)
levels(scenario2.data.group)
