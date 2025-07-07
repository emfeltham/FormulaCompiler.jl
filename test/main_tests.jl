Random.seed!(1234)

@testset "InplaceModeler zero-allocation correctness" begin
    #─ scenario 1: simple linear + categorical + interactions ────────────────
    @testset "Linear + categorical interactions" begin
        n = 600_000
        df = DataFrame(
            y   = randn(n),
            x1  = randn(n),
            x2  = randn(n),
            g   = categorical(rand(["A","B","C"], n)),
            h   = categorical(rand(Bool, n)),
        )
        f  = @formula(y ~ x1 + x2 + g + h + x1 & g + x1 & x2 + g & h)
        m1 = lm(f, df)

        Xref = modelmatrix(m1)
        @test size(Xref) == (n, width(formula(m1).rhs))
        @test eltype(Xref) == Float64

        ipm = InplaceModeler(m1, n)
        X   = similar(Xref)
        fill!(X, 0.0)

        @time modelmatrix!(ipm, Tables.columntable(df), X)
        @test X ≈ Xref
    end

    #─ scenario 1a: simple linear + categorical + interactions + function ──────
    @testset "Linear + categorical interactions" begin
        n = 600_000
        df = DataFrame(
            y   = randn(n),
            x1  = randn(n),
            x2  = randn(n),
            g   = categorical(rand(["A","B","C"], n)),
            h   = categorical(rand(Bool, n)),
        )
        f  = @formula(y ~ x1 + x2 + g + h + x1 & g + x1 & x2 + g & h + x1^2)
        m1 = lm(f, df)

        Xref = modelmatrix(m1)
        @test size(Xref) == (n, width(formula(m1).rhs))
        @test eltype(Xref) == Float64

        ipm = InplaceModeler(m1, n)
        X   = similar(Xref)
        fill!(X, 0.0)

        @time modelmatrix!(ipm, Tables.columntable(df), X)
        @test X ≈ Xref
    end

    #─ scenario 2: nested function terms ──────────────────────────────────────
    @testset "Nested function and arithmetic terms" begin
        n = 1_000
        df = DataFrame(
            y  = randn(n),
            a  = randn(n),
            b  = randn(n),
        )
        # formula with nested functions and arithmetic: a * inv(b) + log(abs.(a - b))
        f2 = @formula(y ~ 0 + a * inv(b) + (abs(a))^2 + (a)^4 & b)
        m2 = lm(f2, df)

        Xref2 = modelmatrix(m2)
        @test size(Xref2) == (n, width(formula(m2).rhs))

        ipm2 = InplaceModeler(m2, n)
        X2   = similar(Xref2)
        fill!(X2, 0.0)

        @time modelmatrix!(ipm2, Tables.columntable(df), X2)
        @test X2 ≈ Xref2
    end

    #─ scenario 3: mixed‐effects model ────────────────────────────────────────
    @testset "MixedModels fixed-effects design" begin
        # smaller dataset for mixed model
        n_small = 5_000
        g_small = repeat(1:50, inner = n_small ÷ 50)
        dfm = DataFrame(
            y  = randn(n_small),
            x  = randn(n_small),
            g  = categorical(g_small),
        )
        # random intercept and slope
        mm = fit(MixedModel, @formula(y ~ 1 + x + (1 + x | g)), dfm)

        # extract the fixed‐effects design matrix only:
        Xref3 = modelmatrix(mm) # MixedModels exports modelmatrix

        fx_fe = fixed_effects_form(mm).rhs;

        @test size(Xref3) == (n_small, width(fx_fe))

        ipm3 = InplaceModeler(mm, n_small)
        X3   = similar(Xref3)
        fill!(X3, 0.0)

        @time modelmatrix!(ipm3, Tables.columntable(dfm), X3)
        @test X3 ≈ Xref3
    end
end
