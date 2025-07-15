# runtests.jl

using Test
using Revise
using EfficientModelMatrices

@testset "EfficientModelMatrices.jl" begin
    include("initial_tests.jl")
    # include("main_tests.jl")
    # include("standardized_tests.jl")
end
