# usage.jl
# Complete Usage Example

using Revise
using Test
using BenchmarkTools
using FormulaCompiler

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
df.cat2a = categorical(rand(["X", "Y"], nrow(df)))
df.cat2b = categorical(rand(["P", "Q"], nrow(df)))
data = Tables.columntable(df);

desc = describe(df);
desc.unique = [unique(df[!, x]) for x in desc.variable];
desc.levels = length.(desc.unique);

@testset "2-level tests" begin
    @testset "cat 2 x cat 2" begin
        i = 1
        model = lm(@formula(y ~ cat2a * cat2b), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end


    @testset "cat 2 x bool" begin
        i = 1
        model = lm(@formula(y ~ cat2a * bool), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end

    @testset "cat 2 x cts" begin
        i = 1
        model = lm(@formula(y ~ cat2a * (x^2)), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end

    @testset "binary x cts" begin
        i = 1
        model = lm(@formula(y ~ bool * (x^2)), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end

    @testset "cat 2 x cts" begin
        i = 1
        model = lm(@formula(y ~ cat2b * (x^2)), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end
end

@testset "3-level tests" begin
    @testset "cat >2 x cts" begin
        i = 1
        model = lm(@formula(y ~ group2 * (x^2)), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end

    @testset "cat >2 x bool" begin
        i = 1
        model = lm(@formula(y ~ group2 * bool), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end

    @testset "cat >2 x cat 2" begin
        i = 1
        model = lm(@formula(y ~ group2 * cat2a), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end

    @testset "cat >2 x cat >2" begin
        i = 1
        model = lm(@formula(y ~ group2 * group3), df)
        fc = compile_formula(model)
        row_vec = Vector{Float64}(undef, fc.output_width);
        modelrow!(row_vec, fc.formula_val, data, i)
        @test modelmatrix(model)[i, :] == row_vec;
    end
end

##

model = lm(@formula(y ~ x + x^2 * log(z) + group), df)

# 2. One-time compilation (expensive, ~1-10ms)
fc = compile_formula(model)

# 3. Setup for fast evaluation
i = 1
row_vec = Vector{Float64}(undef, fc.output_width);
modelrow!(row_vec, fc.formula_val, data, i)
@assert modelmatrix(model)[i, :] == row_vec;

# 4. Zero-allocation runtime usage (~50-100ns per call)
function rowloop!(row_vec, fc, data)
    for i in 1:1000
        modelrow!(row_vec, fc.formula_val, data, i)
        # Now row_vec contains the model matrix row for observation i
        # Use row_vec for predictions, marginal effects, etc.
    end
end

fill!(row_vec, 0.0);
@btime rowloop!(row_vec, fc, data);

# 5. Performance testing
test_compilation_performance(model, data)

##

# 1. Create your data and model

model = lm(@formula(y ~ group2 * group), df)
levels(df.group)
levels(df.group2)

model = lm(@formula(y ~ group2 * group3), df)  # Both have 2 columns each
# This should produce a 4-column interaction (2×2), see if it does

# Create 2-level categoricals
model = lm(@formula(y ~ cat2a * cat2b), df)

data = columntable(df);

# 2. One-time compilation (expensive, ~1-10ms)
fc = compile_formula(model)

# 3. Setup for fast evaluation
i = 1
row_vec = Vector{Float64}(undef, fc.output_width);
modelrow!(row_vec, fc.formula_val, data, i)
@assert isapprox(modelmatrix(model)[i, :], row_vec);

levels(df.group)
levels(df.group2)
levels(df.group3)


println("Row 1 categorical values:")
println("group2: $(data.group2[1]) → level $(levelcode(data.group2[1]))")
println("group3: $(data.group3[1]) → level $(levelcode(data.group3[1]))")

println("\nContrast contributions:")
println("group2 contrasts for level $(levelcode(data.group2[1])): $(modelmatrix(model)[1, 2:3])")
println("group3 contrasts for level $(levelcode(data.group3[1])): $(modelmatrix(model)[1, 4:5])")

println("\nExpected interaction (columns 6-9): $(modelmatrix(model)[1, 6:9])")
println("Your result (columns 6-9): $(row_vec[6:9])")



####

import JLD2
df = JLD2.load_object("/Users/emf/Documents/Yale/yale research/honduras/Honduras CSS/honduras-css-homophily/data/df 2024-06-24.jld2")
df.gender = categorical(df.man_p)
df.coffee = categorical(df.coffee_cultivation)
data = columntable(df);
fx = @formula(response ~ coffee_cultivation * gender)
model = glm(fx, df, Bernoulli(), LogitLink())


# 2. One-time compilation (expensive, ~1-10ms)
fc = compile_formula(model)

i = 1
mm1 = modelmatrix(model)[i, :]
row_vec = Vector{Float64}(undef, fc.output_width);
modelrow!(row_vec, fc.formula_val, data, i)
@assert isapprox(mm1, row_vec);