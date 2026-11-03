# test_threaded_level_cache.jl
# Runs threaded_level_cache_checks.jl in a subprocess with 4 threads, so the concurrency
# test runs regardless of how many threads the test process itself has.

using Test

@testset "LevelIndex pool cache: concurrent evaluation (subprocess, -t 4)" begin
    script = joinpath(@__DIR__, "threaded_level_cache_checks.jl")
    cmd = `$(Base.julia_cmd()) --threads=4 --project=$(Base.active_project()) $script`
    @test success(pipeline(cmd; stdout = stdout, stderr = stderr))
end
