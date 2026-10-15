# test_categorical_level_resolution.jl
# Regression tests for how ContrastOp resolves categorical values to contrast-matrix rows.
#
# Before 1.4.3:
# - A2: the compiled-formula cache ignored categorical pools, and the compiled object baked in
#   a ref-code remap for the compile-time pool. Data with a reordered, subset or rebuilt pool
#   (including a reference grid with its own small pool) reused the stale remap -> wrong rows.
# - A3: non-CategoricalArray categoricals (Vector{String}, integer codes declared categorical
#   via contrasts) re-derived the level index per row from `sort(unique(column))` of the data
#   being evaluated (wrong for subsets and custom level orders; O(n) per row), and integer
#   values were used directly as row indices (wrong unless the codes were exactly 1:k).
# - A6: a missing value (ref code 0) or out-of-range code was clamped to the first row.
# - A8: ContrastEvaluator Bool / integer categorical contrasts returned all zeros (true and
#   false both mapped to row 1).
#
# Oracle throughout: StatsModels' `modelcols(formula(model).rhs, data)`, i.e. the fitted
# schema applied to the evaluation data.

using Test, Random
using DataFrames, Tables, GLM, StatsModels, CategoricalArrays
using FormulaCompiler
using FormulaCompiler: get_or_compile_formula, COMPILED_MODEL_CACHE, ContrastOp

# Max abs difference between compiled rows on `evaldata` and the fitted schema's model matrix
function max_row_error(compiled, model, evaldata)
    X = modelcols(StatsModels.formula(model).rhs, evaldata)
    out = zeros(length(compiled))
    err = 0.0
    for i in 1:size(X, 1)
        compiled(out, evaldata, i)
        err = max(err, maximum(abs.(out .- X[i, :])))
    end
    return err
end

function row_allocations(compiled, data, row)
    out = zeros(length(compiled))
    compiled(out, data, row)
    return @allocated compiled(out, data, row)
end

@testset "Categorical level resolution" begin
    Random.seed!(20261015)
    n = 60
    s = rand(["a", "b", "c"], n)
    df = DataFrame(y = randn(n), x = randn(n), s = s, g = categorical(s),
                   i01 = rand(0:1, n), i10 = rand([10, 20, 30], n), b = rand(Bool, n))
    d = Tables.columntable(df)

    @testset "A3: String and integer-coded categoricals use the model's levels" begin
        cases = [
            ("String, default", @formula(y ~ x + s), Dict{Symbol,Any}()),
            ("String, DummyCoding(levels=[c,a,b])", @formula(y ~ x + s),
             Dict{Symbol,Any}(:s => DummyCoding(levels = ["c", "a", "b"]))),
            ("String, EffectsCoding(base=b) & x", @formula(y ~ x * s),
             Dict{Symbol,Any}(:s => EffectsCoding(base = "b"))),
            ("Int 0/1 categorical", @formula(y ~ x + i01), Dict{Symbol,Any}(:i01 => DummyCoding())),
            ("Int 10/20/30 categorical", @formula(y ~ x + i10), Dict{Symbol,Any}(:i10 => DummyCoding())),
            ("Int 10/20/30, levels=[30,10,20]", @formula(y ~ x + i10),
             Dict{Symbol,Any}(:i10 => DummyCoding(levels = [30, 10, 20]))),
            ("Bool categorical", @formula(y ~ x + b), Dict{Symbol,Any}(:b => DummyCoding())),
            ("Categorical, levels=[c,a,b]", @formula(y ~ x + g),
             Dict{Symbol,Any}(:g => DummyCoding(levels = ["c", "a", "b"]))),
        ]
        for (label, f, contrasts) in cases
            m = lm(f, df; contrasts = contrasts)
            c = compile_formula(m, d)
            @test max_row_error(c, m, d) < 1e-12
        end

        # Evaluation data lacking a level ("b") must keep the fitted coding
        m = lm(@formula(y ~ x + s), df)
        sub = Tables.columntable(df[df.s .!= "b", :])
        @test max_row_error(compile_formula(m, d), m, sub) < 1e-12
        @test max_row_error(compile_formula(m, sub), m, sub) < 1e-12

        mi = lm(@formula(y ~ x + i10), df; contrasts = Dict(:i10 => DummyCoding()))
        subi = Tables.columntable(df[df.i10 .!= 10, :])
        @test max_row_error(compile_formula(mi, subi), mi, subi) < 1e-12

        # Values that are not model levels throw instead of picking a row
        c = compile_formula(m, d)
        out = zeros(length(c))
        bad = merge(d, (s = fill("zzz", n),))
        @test_throws ArgumentError c(out, bad, 1)
        ci = compile_formula(mi, d)
        badi = merge(d, (i10 = fill(40, n),))
        @test_throws ArgumentError ci(zeros(length(ci)), badi, 1)

        # O(1), allocation-free per row for String columns (was O(n) with 400 B per row)
        big_n = 20_000
        sb = rand(["a", "b", "c"], big_n)
        dfb = DataFrame(y = randn(big_n), x = randn(big_n), s = sb)
        mb = lm(@formula(y ~ x * s), dfb)
        db = Tables.columntable(dfb)
        cb = compile_formula(mb, db)
        @test row_allocations(cb, db, 17) == 0
        out = zeros(length(cb))
        cb(out, db, 1)
        t = @elapsed for i in 1:big_n
            cb(out, db, i)
        end
        @test t < 0.5   # 20K rows; the old O(n)-per-row path took ~2 s here
    end

    @testset "A2: compiled objects do not depend on the evaluation data's pool" begin
        m = lm(@formula(y ~ x + g), df)
        X = modelmatrix(m)
        empty!(COMPILED_MODEL_CACHE)
        modelrow(m, d, 1)                                    # populate the cache

        # Same data: still a cache hit
        @test get_or_compile_formula(m, d) === get_or_compile_formula(m, d)

        # Subset with a rebuilt pool lacking "b"
        sub = df[df.g .!= "b", :]
        sub.g = droplevels!(copy(sub.g))
        @test levels(sub.g) == ["a", "c"]
        dsub = Tables.columntable(sub)
        truth = modelcols(StatsModels.formula(m).rhs, dsub)
        rows = collect(1:nrow(sub))
        @test maximum(abs.(modelrow(m, dsub, rows) .- truth)) < 1e-12
        @test maximum(abs.(modelrow(compile_formula(m, dsub), dsub, rows) .- truth)) < 1e-12

        # Pool reordered in place
        df3 = copy(df)
        df3.g = copy(df.g)
        levels!(df3.g, ["c", "b", "a"])
        d3 = Tables.columntable(df3)
        @test maximum(abs.(modelrow(m, d3, collect(1:n)) .- X)) < 1e-12

        # Reference grid with its own one-level pool, evaluated with an object compiled on
        # the full data (directly and through the cache)
        grid = (x = [0.0], g = categorical(["c"]))
        c_full = compile_formula(m, d)
        gtruth = modelcols(StatsModels.formula(m).rhs, grid)
        out = zeros(length(c_full))
        c_full(out, grid, 1)
        @test maximum(abs.(out .- vec(gtruth))) < 1e-12
        modelrow(m, (x = zeros(3), g = categorical(["a", "b", "c"])), 1)
        @test maximum(abs.(modelrow(m, grid, 1) .- vec(gtruth))) < 1e-12

        # Alternating pools keeps resolving correctly, and the compile-time pool stays
        # allocation-free
        for _ in 1:3
            c_full(out, grid, 1)
            @test out ≈ vec(gtruth)
            c_full(out, d, 5)
            @test out ≈ X[5, :]
        end
        c_fresh = compile_formula(m, d)
        @test row_allocations(c_fresh, d, 3) == 0

        # Levels appended to the shared pool after compilation (codes of existing levels
        # are unchanged); a row carrying the new level is not a model level -> error
        g2 = copy(df.g)
        c2 = compile_formula(m, merge(d, (g = g2,)))
        d2 = merge(d, (g = g2,))
        levels!(g2, ["a", "b", "c", "d"])
        @test max_row_error(c2, m, d2) < 1e-12
        g2[1] = "d"
        @test_throws ArgumentError c2(zeros(length(c2)), d2, 1)

        # Categorical data for a model fitted on a String column, and vice versa
        ms = lm(@formula(y ~ x + s), df)
        dcat = merge(d, (s = categorical(df.s; levels = ["c", "b", "a"]),))
        @test max_row_error(compile_formula(ms, d), ms, dcat) < 1e-12
        dstr = merge(d, (g = String.(df.s),))
        @test max_row_error(compile_formula(m, d), m, dstr) < 1e-12
    end

    @testset "A6: missing and out-of-range codes error instead of clamping" begin
        m = lm(@formula(y ~ x + g), df)
        c = compile_formula(m, d)
        out = zeros(length(c))
        gm = CategoricalArray{Union{Missing,String}}(df.s)
        gm[2] = missing
        dm = merge(d, (g = gm,))
        @test_throws ArgumentError c(out, dm, 2)
        c(out, dm, 1)                                        # rows without missing still work
        @test out ≈ modelmatrix(m)[1, :]

        # Counterfactual vector over a base column with a missing ref code
        cf = FormulaCompiler.counterfactualvector(gm, 0)
        @test cf isa FormulaCompiler.CategoricalCounterfactualVector
        @test_throws ArgumentError c(out, merge(d, (g = cf,)), 2)

        # Positional ContrastOp (bare matrix): out-of-range integer codes throw
        op = ContrastOp{:k, (1, 2)}([1.0 0.0; 0.0 1.0; -1.0 -1.0])
        scratch = zeros(2)
        FormulaCompiler.execute_op(op, scratch, (k = [3],), 1)
        @test scratch == [-1.0, -1.0]
        @test_throws ArgumentError FormulaCompiler.execute_op(op, scratch, (k = [4],), 1)
        @test_throws ArgumentError FormulaCompiler.execute_op(op, scratch, (k = [0],), 1)
    end

    @testset "A8: ContrastEvaluator with Bool and integer categoricals" begin
        mb = lm(@formula(y ~ x + b), df; contrasts = Dict(:b => DummyCoding()))
        cb = compile_formula(mb, d)
        ceb = contrastevaluator(cb, d, [:b])
        @test contrast_modelrow(ceb, 1, :b, false, true) == [0.0, 0.0, 1.0]
        @test contrast_modelrow(ceb, 1, :b, true, false) == [0.0, 0.0, -1.0]
        Δ = zeros(length(cb))
        contrast_modelrow!(Δ, ceb, 2, :b, false, true)
        @test (@allocated contrast_modelrow!(Δ, ceb, 2, :b, false, true)) == 0
        β = coef(mb)
        @test contrast_gradient(ceb, 1, :b, false, true, β) == [0.0, 0.0, 1.0]

        m01 = lm(@formula(y ~ x + i01), df; contrasts = Dict(:i01 => DummyCoding()))
        c01 = compile_formula(m01, d)
        ce01 = contrastevaluator(c01, d, [:i01])
        @test contrast_modelrow(ce01, 1, :i01, 0, 1) == [0.0, 0.0, 1.0]

        # Integer codes that are not 1:k, with a non-default base level
        m10 = lm(@formula(y ~ x + i10), df; contrasts = Dict(:i10 => EffectsCoding(base = 20)))
        c10 = compile_formula(m10, d)
        ce10 = contrastevaluator(c10, d, [:i10])
        lev_row(v) = vec(modelcols(StatsModels.formula(m10).rhs, (x = [0.0], i10 = [v])))
        @test contrast_modelrow(ce10, 1, :i10, 10, 30) ≈ lev_row(30) .- lev_row(10)
        @test contrast_modelrow(ce10, 1, :i10, 20, 10) ≈ lev_row(10) .- lev_row(20)
        @test_throws ArgumentError contrast_modelrow(ce10, 1, :i10, 10, 40)

        # Categorical data still contrasts correctly (unchanged path)
        mg = lm(@formula(y ~ x + g), df)
        cg = compile_formula(mg, d)
        ceg = contrastevaluator(cg, d, [:g])
        @test contrast_modelrow(ceg, 1, :g, "a", "c") == [0.0, 0.0, 0.0, 1.0]
        Δg = zeros(length(cg))
        contrast_modelrow!(Δg, ceg, 2, :g, "a", "b")
        @test (@allocated contrast_modelrow!(Δg, ceg, 2, :g, "a", "b")) == 0
    end

    # Model-based categorical queries, and contrasts targeting levels absent from the data's
    # pool. Before: an integer column declared categorical had no level map in the
    # ContrastEvaluator, and a pool lacking the model's reference level could not express
    # contrasts against it ("Baseline level not found"; Margins padded pools as a workaround).
    @testset "Model levels: categorical_levels and contrasts to absent pool levels" begin
        rng = Random.MersenneTwister(11)
        n = 400
        df = DataFrame(x = randn(rng, n), k = rand(rng, 1:3, n)); df.y = randn(rng, n)
        m = lm(@formula(y ~ x * k), df; contrasts = Dict(:k => DummyCoding()))
        d = Tables.columntable(df)
        c = compile_formula(m, d)
        @test categorical_levels(c) == Dict(:k => [1, 2, 3])
        @test categorical_variables(c) == [:k]
        @test continuous_variables(c, d) == [:x]
        ce = contrastevaluator(c, d, [:k])
        buf = zeros(length(c))
        for row in (1, 7), (a, b) in ((1, 3), (2, 3))
            contrast_modelrow!(buf, ce, row, :k, a, b)
            lo = copy(df[row:row, :]); hi = copy(lo); lo.k .= a; hi.k .= b
            @test buf ≈ vec(modelcols(formula(m).rhs, hi) .- modelcols(formula(m).rhs, lo))
        end

        df2 = DataFrame(x = randn(rng, n), g = categorical(rand(rng, ["a", "b", "c"], n)))
        df2.y = randn(rng, n)
        m2 = lm(@formula(y ~ x * g), df2)
        sub = df2[df2.g .!= "a", :]
        sub.g = categorical(String.(sub.g))                 # pool: "b", "c" (no reference level)
        ds = Tables.columntable(sub)
        c2 = compile_formula(m2, ds)
        ce2 = contrastevaluator(c2, ds, [:g])
        buf2 = zeros(length(c2))
        for row in (1, 5)
            contrast_modelrow!(buf2, ce2, row, :g, "a", "c")
            lo = DataFrame(x = [sub.x[row]], g = categorical(["a"]; levels = ["a", "b", "c"]), y = [0.0])
            hi = copy(lo); hi.g .= "c"
            @test buf2 ≈ vec(modelcols(formula(m2).rhs, hi) .- modelcols(formula(m2).rhs, lo))
        end
        @test levels(sub.g) == ["b", "c"]                   # caller's column untouched
        contrast_modelrow!(buf2, ce2, 3, :g, "a", "b")
        @test (@allocated contrast_modelrow!(buf2, ce2, 3, :g, "a", "b")) == 0
    end
end
