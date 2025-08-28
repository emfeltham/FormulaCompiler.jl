# test/bootstrap.jl — Convenience bootstrap for running a single test file

try
    @eval using FormulaCompiler
catch err
    @info "Developing local FormulaCompiler for test project" exception=(err, catch_backtrace())
    import Pkg
    Pkg.develop(Pkg.PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
    @eval using FormulaCompiler
end

using Test
using Random
using LinearAlgebra
using Statistics
import Pkg

"""
    _fc_load_testdeps(; csv=false)

Load common optional test dependencies. Call this from a test file after
including `bootstrap.jl` when running a single file directly.

Examples
    include("test/bootstrap.jl"); _fc_load_testdeps()
    include("test/bootstrap.jl"); _fc_load_testdeps(csv=true)
"""
function _fc_load_testdeps(; csv::Bool=false)
    # BenchmarkTools
    try
        @eval using BenchmarkTools
    catch
        try
            Pkg.instantiate(); @eval using BenchmarkTools
        catch
            @warn "BenchmarkTools not available; allocation tests may not run"
        end
    end
    # CSV (optional)
    if csv
        try
            @eval using CSV
        catch
            try
                Pkg.instantiate(); @eval using CSV
            catch
                @warn "CSV not available; CSV output will be skipped"
            end
        end
    end
    return nothing
end

