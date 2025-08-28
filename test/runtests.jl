# test/runtests.jl
# Main test runner for FormulaCompiler.jl
# julia --project="test" test/runtests.jl > test/tests.txt 2>&1

using Test
using Random

# Ensure local package is available when running with --project=test
try
    @eval using FormulaCompiler
catch err
    @info "Developing local FormulaCompiler for test project" exception=(err, catch_backtrace())
    import Pkg
    Pkg.develop(Pkg.PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
    @eval using FormulaCompiler
end

# Reproducible tests
Random.seed!(06515)

# Include test support utilities (moved out of the package module)
include(joinpath(@__DIR__, "support", "testing_utilities.jl"))

@testset "FormulaCompiler.jl Tests" begin    
    # Core functionality
    include("test_position_mapping.jl") # Position mapping system
    
    # Models
    include("test_allocations.jl") # Performance
    include("test_models.jl") # Correctness

    # Override and scenario system
    include("test_overrides.jl") # Override and scenario functionality
    include("test_categorical_correctness.jl") # Detailed categorical override correctness
    
    # Derivatives
    include("test_derivatives.jl")
    include("test_links.jl")
    include("test_derivative_allocations.jl")
end
