using Test
using Random
using DataFrames, CategoricalArrays
using Margins

if Base.find_package("MixedModels") !== nothing
    using MixedModels
    @testset "MixedModels fixed-effects sanity" begin
        Random.seed!(5)
        n = 200
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            grp = categorical(rand(1:10, n))
        )
        m = fit(MixedModels.LinearMixedModel, @formula(y ~ 1 + x + (1|grp)), df)
        # Expect single-row AME for fixed effect x
        res = ame(m, df; dydx=[:x])
        @test nrow(res.table) == 1
        @test all(isfinite, res.table.dydx)
    end
end

