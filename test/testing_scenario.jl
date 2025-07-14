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