# test_compiled_cache.jl
# FC 1.5.0 (M1): the compiled-formula cache behind `get_or_compile_formula` is bounded (LRU),
# can be cleared, releases the models it drops, and is safe under concurrent access.
# Before 1.5.0 it was an unbounded Dict that kept every model alive for the whole session.

using Test, Random
using DataFrames, Tables, GLM, CategoricalArrays
using FormulaCompiler

function _cache_test_model(rng, n = 60)
    df = DataFrame(y = randn(rng, n), x = randn(rng, n),
                   g = categorical(rand(rng, ["a", "b", "c"], n)))
    return lm(@formula(y ~ x * g), df), Tables.columntable(df)
end

# Fit a model, compile it through the cache, and return only a weak reference to one of
# the model's arrays, so the caller holds no strong reference to the model.
function _cached_model_weakref(rng)
    m, d = _cache_test_model(rng)
    get_or_compile_formula(m, d)
    return WeakRef(modelmatrix(m))   # an array owned by the model
end

_hit_allocations(m, d) = (get_or_compile_formula(m, d); @allocated get_or_compile_formula(m, d))

@testset "Compiled-formula cache (bounded, clearable)" begin
    rng = MersenneTwister(1016)
    old_capacity = compiled_cache_capacity()
    try
        clear_compiled_cache!()
        set_compiled_cache_capacity!(16)

        @testset "API" begin
            @test compiled_cache_capacity() == 16
            @test set_compiled_cache_capacity!(5) == 16
            @test compiled_cache_capacity() == 5
            @test_throws ArgumentError set_compiled_cache_capacity!(-1)
            @test clear_compiled_cache!() === nothing
            @test FormulaCompiler.cache_size() == 0
            @test FormulaCompiler.cache_info() == (entries = 0, capacity = 5)
            set_compiled_cache_capacity!(16)
        end

        @testset "repeated calls hit the cache" begin
            clear_compiled_cache!()
            m, d = _cache_test_model(rng)
            c1 = get_or_compile_formula(m, d)
            c2 = get_or_compile_formula(m, d)
            @test c1 === c2
            @test FormulaCompiler.cache_size() == 1
            # modelrow and modelrow! with a model go through the same cache
            @test modelrow(m, d, 3) == modelmatrix(m)[3, :]
            out = zeros(length(c1)); modelrow!(out, m, d, 4)
            @test out == modelmatrix(m)[4, :]
            @test FormulaCompiler.cache_size() == 1
            # same columns, Int instead of Float64 values: same semantic key
            d_int = merge(d, (x = round.(Int, 10 .* d.x),))
            @test get_or_compile_formula(m, d_int) === c1
            # a hit allocates nothing
            @test _hit_allocations(m, d) == 0
        end

        @testset "1.5.0 level semantics: pools are not part of the key" begin
            clear_compiled_cache!()
            m, d = _cache_test_model(rng)
            c = get_or_compile_formula(m, d)
            # reordered pool, and a reference grid with its own small pool
            g2 = categorical(String.(d.g); levels = ["c", "a", "b"])
            d2 = merge(d, (g = g2,))
            @test get_or_compile_formula(m, d2) === c
            @test modelrow(m, d2, 5) == modelmatrix(m)[5, :]
            grid = (y = [0.0], x = [0.5], g = categorical(["b"]))
            @test get_or_compile_formula(m, grid) === c
            # a column changing category (categorical -> String) is a different key
            d3 = merge(d, (g = String.(d.g),))
            c3 = get_or_compile_formula(m, d3)
            @test c3 !== c
            @test modelrow(m, d3, 7) == modelmatrix(m)[7, :]
        end

        @testset "mixtures are read from the data, not baked into the cached object" begin
            # Before the fix the compiled MixtureContrastOp embedded the compile-time mixture's
            # levels and weights in its type, while the key only records that the column is a
            # mixture: a second grid with another mixture got the first grid's model row.
            clear_compiled_cache!()
            m, _ = _cache_test_model(rng)
            # y ~ x * g with g ∈ {a, b, c} (DummyCoding, base a): [1, x, b, c, x*b, x*c]
            hand(x, s) = (sb = get(s, "b", 0.0); sc = get(s, "c", 0.0); [1, x, sb, sc, x * sb, x * sc])
            grid(x, mx) = (y = [0.0], x = [x], g = [mx])
            mixA = FormulaCompiler.mix("a" => 0.5, "b" => 0.25, "c" => 0.25)
            mixW = FormulaCompiler.mix("a" => 0.1, "b" => 0.1, "c" => 0.8)       # other weights
            mixL = FormulaCompiler.mix("c" => 0.6, "b" => 0.4)                     # other levels/order
            cA = get_or_compile_formula(m, grid(0.7, mixA))
            @test modelrow(m, grid(0.7, mixA), 1) ≈ hand(0.7, Dict("a" => 0.5, "b" => 0.25, "c" => 0.25))
            # same key (the model and the column categories match) ...
            @test get_or_compile_formula(m, grid(0.7, mixW)) === cA
            @test get_or_compile_formula(m, grid(0.7, mixL)) === cA
            # ... and each evaluates its own mixture
            @test modelrow(m, grid(0.7, mixW), 1) ≈ hand(0.7, Dict("a" => 0.1, "b" => 0.1, "c" => 0.8))
            @test modelrow(m, grid(-1.3, mixL), 1) ≈ hand(-1.3, Dict("c" => 0.6, "b" => 0.4))
            @test modelrow(m, grid(0.7, mixA), 1) ≈ hand(0.7, Dict("a" => 0.5, "b" => 0.25, "c" => 0.25))
            # a plain level after a mixture (a different key)
            @test modelrow(m, (y = [0.0], x = [0.7], g = categorical(["c"])), 1) ≈ hand(0.7, Dict("c" => 1.0))
            # untyped columns: a mixture, then a plain level, in the same `Vector{Any}` column type
            @test modelrow(m, (y = [0.0], x = [0.7], g = Any[mixW]), 1) ≈ hand(0.7, Dict("a" => 0.1, "b" => 0.1, "c" => 0.8))
            @test modelrow(m, (y = [0.0], x = [0.7], g = Any["b"]), 1) ≈ hand(0.7, Dict("b" => 1.0))
            # a different mixture in each row of one column
            d2 = (y = [0.0, 0.0], x = [0.7, 0.7], g = [mixA, mixL])
            @test modelrow(m, d2, 2) ≈ hand(0.7, Dict("c" => 0.6, "b" => 0.4))
            # a mixture edited in place is not resolved with the old level map
            mixE = FormulaCompiler.mix("a" => 0.5, "b" => 0.5)
            out = zeros(length(cA))
            cA(out, grid(0.7, mixE), 1)
            @test out ≈ hand(0.7, Dict("b" => 0.5))
            mixE.levels[2] = "c"; mixE.weights .= (0.25, 0.75)
            cA(out, grid(0.7, mixE), 1)
            @test out ≈ hand(0.7, Dict("c" => 0.75))
            # zero allocations on a column of one repeated mixture
            dfill = (y = zeros(20), x = fill(0.7, 20), g = fill(mixW, 20))
            c = FormulaCompiler.with_private_scratch(get_or_compile_formula(m, dfill))
            c(out, dfill, 1)
            @test (@allocated c(out, dfill, 7)) == 0
            @test out ≈ hand(0.7, Dict("a" => 0.1, "b" => 0.1, "c" => 0.8))
            # a contrast between two mixtures on a mixture column
            dmix = (y = zeros(3), x = fill(0.7, 3), g = fill(mixA, 3))
            ev = contrastevaluator(compile_formula(m, dmix), dmix, [:g])
            Δ = zeros(length(cA))
            contrast_modelrow!(Δ, ev, 1, :g, mixA, mixW)
            @test Δ ≈ hand(0.7, Dict("a" => 0.1, "b" => 0.1, "c" => 0.8)) .-
                     hand(0.7, Dict("a" => 0.5, "b" => 0.25, "c" => 0.25))
        end

        @testset "LRU eviction" begin
            clear_compiled_cache!()
            set_compiled_cache_capacity!(3)
            md = [_cache_test_model(rng) for _ in 1:4]
            cs = [get_or_compile_formula(m, d) for (m, d) in md[1:3]]
            @test FormulaCompiler.cache_size() == 3
            get_or_compile_formula(md[1]...)            # 1 is now the most recent; 2 the oldest
            c4 = get_or_compile_formula(md[4]...)       # evicts 2
            @test FormulaCompiler.cache_size() == 3
            @test get_or_compile_formula(md[1]...) === cs[1]
            @test get_or_compile_formula(md[3]...) === cs[3]
            @test get_or_compile_formula(md[4]...) === c4
            c2b = get_or_compile_formula(md[2]...)      # recompiled after eviction
            @test c2b !== cs[2]
            @test modelrow(c2b, md[2][2], 2) == modelmatrix(md[2][1])[2, :]
            # lowering the capacity evicts immediately
            @test set_compiled_cache_capacity!(1) == 3
            @test FormulaCompiler.cache_size() == 1
            @test get_or_compile_formula(md[2]...) === c2b   # most recently used survives
            set_compiled_cache_capacity!(16)
        end

        @testset "capacity 0 disables caching" begin
            clear_compiled_cache!()
            set_compiled_cache_capacity!(0)
            m, d = _cache_test_model(rng)
            c1 = get_or_compile_formula(m, d)
            c2 = get_or_compile_formula(m, d)
            @test c1 !== c2
            @test FormulaCompiler.cache_size() == 0
            @test modelrow(m, d, 1) == modelmatrix(m)[1, :]
            set_compiled_cache_capacity!(16)
        end

        @testset "models are released (clear and eviction)" begin
            clear_compiled_cache!()
            set_compiled_cache_capacity!(16)
            refs = [_cached_model_weakref(rng) for _ in 1:3]
            GC.gc(); GC.gc()
            @test all(r -> r.value !== nothing, refs)    # held by the cache
            compiled_survivor = get_or_compile_formula(_cache_test_model(rng)...)
            clear_compiled_cache!()
            GC.gc(); GC.gc()
            @test all(r -> r.value === nothing, refs)    # released
            @test length(compiled_survivor) > 0          # handed-out objects stay valid

            # A bootstrap-like loop keeps at most `capacity` models alive
            set_compiled_cache_capacity!(4)
            refs = [_cached_model_weakref(rng) for _ in 1:20]
            GC.gc(); GC.gc()
            @test FormulaCompiler.cache_size() == 4
            @test count(r -> r.value !== nothing, refs) <= 4
            @test all(r -> r.value !== nothing, refs[17:20])
            set_compiled_cache_capacity!(16)
        end

        @testset "concurrent access does not corrupt the cache" begin
            clear_compiled_cache!()
            set_compiled_cache_capacity!(3)
            md = [_cache_test_model(rng) for _ in 1:6]
            expected = [modelmatrix(m)[2, :] for (m, _) in md]
            ok = Threads.Atomic{Int}(0)
            @sync for t in 1:48
                Threads.@spawn begin
                    i = mod1(t, 6)
                    m, d = md[i]
                    c = get_or_compile_formula(m, d)
                    # each task evaluates its own buffer; a cached object is not shared
                    # across concurrent evaluations here (it is not thread-safe)
                    c_own = compile_formula(m, d)
                    c_own === c && error("unexpected identity")
                    modelrow(c_own, d, 2) == expected[i] && Threads.atomic_add!(ok, 1)
                    t % 7 == 0 && clear_compiled_cache!()
                end
            end
            @test ok[] == 48
            @test FormulaCompiler.cache_size() <= 3
            for (i, (m, d)) in enumerate(md)
                @test modelrow(get_or_compile_formula(m, d), d, 2) == expected[i]
            end
        end
    finally
        set_compiled_cache_capacity!(old_capacity)
        clear_compiled_cache!()
    end
end
