# test/runtests.jl
# Main test runner for FormulaCompiler.jl
# julia --project="test" test/runtests.jl > test/tests.txt 2>&1
# julia --project="." -e "import Pkg; Pkg.test()" > test/runtests_2.txt 2>&1

using Test
using Random
using FormulaCompiler

# Reproducible tests
Random.seed!(06515)

# Include test support utilities (moved out of the package module)
include(joinpath(@__DIR__, "support", "testing_utilities.jl"))
include(joinpath(@__DIR__, "support", "generate_large_synthetic_data.jl"))

@testset "FormulaCompiler.jl Tests" begin    
    # Core functionality
    include("test_position_mapping.jl") # Position mapping system
    
    # Models
    include("test_allocations.jl") # PRIMARY: Zero-allocation verification for model compilation
    include("test_models.jl") # PRIMARY: Correctness of compiled formulas
    include("test_logic.jl") # Logic operators (comparisons and boolean negation)
    include("test_tough_formula.jl") # Complex formula compilation test

    # Counterfactual system (scenario analysis with variable substitution)
    include("test_overrides.jl") # Counterfactual vector and scenario functionality
    include("test_zero_allocation_overrides.jl") # Zero-allocation counterfactual performance validation
    include("test_categorical_correctness.jl") # Detailed categorical counterfactual correctness
    include("test_compressed_categoricals.jl") # Compressed categorical arrays (UInt8, UInt16, UInt32)

    # Categorical mixtures (Phase 5 complete implementation)
    include("test_mixture_evaluation.jl") # Comprehensive mixture evaluation primitives
    include("test_mixture_modelrows.jl") # Modelrow correctness with mixtures

    # Derivatives (primitives only - statistical interface migrated to Margins.jl)
    # MIGRATED: test_links.jl → Margins/test/primitives/test_links.jl
    # SPLIT: test_derivative_allocations.jl → test_derivative_primitive_allocations.jl (FormulaCompiler) + test_marginal_effects_allocations.jl (Margins)
    include("test_derivative_primitive_allocations.jl") # PRIMARY: derivative_modelrow! allocation tests (primitives only)
    include("test_contrast_evaluator.jl") # Zero-allocation contrast evaluator correctness and performance
    # DELETED: test_documentation_examples.jl → Caused circular dependency with Margins (2025-10-09)

    # AD allocation validation - Complex formula and batch scaling tests
    include("test_ad_alloc_formula_variants.jl") # Formula pattern allocation profiling
    include("test_formulacompiler_primitives_allocations.jl") # Batch scaling + NamedTuple regression guards

    # Regression tests
    # MIGRATED: test_derivatives_log_profile_regression.jl → Margins/test/primitives/test_derivatives_log_profile_regression.jl
    # Edge-case regression and stability tests
    include("test_derivatives_domain_edge_cases.jl")
    
    # External package integration
    include("test_standardized_predictors.jl") # StandardizedPredictors.jl integration
    include("test_glm_integration.jl") # GLM integration testing
    include("test_mixedmodels_integration.jl") # MixedModels integration testing

    # Performance testing
    include("test_large_dataset_performance.jl") # Large dataset performance validation
end
