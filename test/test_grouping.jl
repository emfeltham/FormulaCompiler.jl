using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Grouping: over/within/by" begin
    Random.seed!(7)
    n = 240
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = randn(n),
        g = categorical(rand(["A","B"], n)),
        h = categorical(rand(["U","V","W"], n))
    )
    m = lm(@formula(y ~ x + z + g + h), df)

    # over by single factor
    over_g = ame(m, df; dydx=[:x], over=:g)
    @test nrow(over_g.table) == length(levels(df.g))
    @test haskey(over_g.table, :g)

    # nested within designs (group by g and within h)
    nested = ame(m, df; dydx=[:x], over=:g, within=:h)
    @test nrow(nested.table) == length(levels(df.g)) * length(levels(df.h))
    @test haskey(nested.table, :g) && haskey(nested.table, :h)

    # by stratification: split AME by g (separate computations)
    by_g = ame(m, df; dydx=[:x], by=:g)
    @test nrow(by_g.table) == length(levels(df.g))
    @test haskey(by_g.table, :g)
end

