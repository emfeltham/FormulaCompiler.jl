# threaded_level_cache_checks.jl
# Run with several threads (see test_threaded_level_cache.jl). Concurrent evaluations that
# share one compiled formula's operations, each with its own scratch buffer, over datasets
# whose categorical columns have *different pool objects*, so the per-ContrastOp
# LevelIndex pool cache is rebuilt under contention. Before the cache was published
# atomically, readers could see a torn (pool, rows) pair: BoundsErrors, or a wrong row.

using Test, Random
using DataFrames, Tables, GLM, CategoricalArrays
using FormulaCompiler

private_copy(c::FormulaCompiler.UnifiedCompiled{T, Ops, S, O}) where {T, Ops, S, O} =
    FormulaCompiler.UnifiedCompiled{T, Ops, S, O}(c.ops)

@testset "LevelIndex pool cache under concurrency (nthreads = $(Threads.nthreads()))" begin
    @test Threads.nthreads() > 1
    rng = MersenneTwister(2026)
    n = 400
    df = DataFrame(x = randn(rng, n), g = categorical(rand(rng, ["a", "b", "c", "d"], n)))
    df.y = randn(rng, n)
    model = lm(@formula(y ~ x * g), df)
    compiled = compile_formula(model, Tables.columntable(df))

    # 8 datasets, each with its own pool: rebuilt pools, reordered levels, smaller pools
    datasets = map(1:8) do i
        d = df[(i - 1) * 50 + 1:i * 50, :]
        vals = String.(d.g)
        lev = i % 3 == 0 ? reverse(sort(unique(vals))) : sort(unique(vals))
        d.g = categorical(vals; levels = lev)
        Tables.columntable(d)
    end
    # Serial references from the fitted model's own design matrix logic
    refs = map(datasets) do d
        reduce(vcat, (modelrow(compiled, d, r)' for r in 1:50))
    end

    for mode in (:spawn, :threads)
        n_jobs = 400
        mismatches = Threads.Atomic{Int}(0)
        errors = Vector{Any}(nothing, n_jobs)
        body = function (j)
            k = mod1(j, length(datasets))
            c = private_copy(compiled)
            out = Vector{Float64}(undef, length(c))
            try
                for _ in 1:20, r in 1:50
                    c(out, datasets[k], r)
                    out == view(refs[k], r, :) || Threads.atomic_add!(mismatches, 1)
                end
            catch e
                errors[j] = e
            end
        end
        if mode === :spawn
            foreach(wait, [Threads.@spawn body(j) for j in 1:n_jobs])
        else
            Threads.@threads for j in 1:n_jobs
                body(j)
            end
        end
        errs = filter(!isnothing, errors)
        isempty(errs) || showerror(stderr, first(errs))
        @test isempty(errs)
        @test mismatches[] == 0
    end
end
