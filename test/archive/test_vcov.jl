using Test
using Random
using DataFrames, GLM
using Margins

@testset "vcov keyword behavior" begin
    Random.seed!(99)
    n = 150
    df = DataFrame(y = randn(n), x = randn(n))
    m = lm(@formula(y ~ x), df)

    base = ame(m, df; dydx=[:x])
    Σ = vcov(m)
    # Inflate covariance to increase SE
    big = ame(m, df; dydx=[:x], vcov=2.0 .* Σ)
    @test base.table.dydx[1] == big.table.dydx[1]
    @test big.table.se[1] > base.table.se[1]

    # Function override equals matrix override
    fun = ame(m, df; dydx=[:x], vcov = m->vcov(m))
    @test isapprox(fun.table.se[1], base.table.se[1]; rtol=1e-12)
end

