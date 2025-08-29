using Test
using Random
using DataFrames, GLM
using Margins

@testset "Errors and invalid inputs" begin
    Random.seed!(1)
    df = DataFrame(y = randn(50), x = randn(50))
    m = lm(@formula(y ~ x), df)

    # Nonexistent variable in dydx
    @test_throws Exception ame(m, df; dydx=[:not_a_var])

    # Invalid vcov spec (estimator object that isn't supported)
    @test_throws Exception ame(m, df; dydx=[:x], vcov=:invalid_estimator)
end

