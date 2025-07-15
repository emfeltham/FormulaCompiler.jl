# runtests.jl

using Test
using Revise
using Random
using EfficientModelMatrices

@testset "EfficientModelMatrices.jl" begin
    include("initial_tests.jl")
    # include("benchmark_tests.jl")
    # OLD:
    # include("main_tests.jl")
    # include("standardized_tests.jl")
end
