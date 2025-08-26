# test/runtests.jl
# Main test runner for FormulaCompiler.jl
# julia --project="." test/runtests.jl > test/tests.txt 2>&1

using Test
using Random

# Reproducible tests
Random.seed!(06515)

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
    include("test_derivatives_extended.jl")
    include("test_links.jl")
    include("test_derivative_allocations.jl")
end
