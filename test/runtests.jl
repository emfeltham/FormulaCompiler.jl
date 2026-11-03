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
    include("test_categorical_level_resolution.jl") # Level resolution vs model levels (pools, String/Int/Bool, missing)

    # Categorical mixtures
    include("test_mixture_evaluation.jl") # Comprehensive mixture evaluation primitives
    include("test_mixture_api.jl") # mix, validate_mixture_against_data, mixture_to_scenario_value
    include("test_mixture_modelrows.jl") # Modelrow correctness with mixtures

    # Derivatives (primitives only - statistical interface migrated to Margins.jl)
    # MIGRATED: test_links.jl → Margins/test/primitives/test_links.jl
    include("test_derivative_primitive_allocations.jl") # PRIMARY: derivative_modelrow! allocation tests (primitives only)
    include("test_contrast_evaluator.jl") # Zero-allocation contrast evaluator correctness and performance
    # DELETED: test_documentation_examples.jl → Caused circular dependency with Margins (2025-10-09)

    # AD allocation validation - Complex formula and batch scaling tests
    include("test_ad_alloc_formula_variants.jl") # Formula pattern allocation profiling
    include("test_formulacompiler_primitives_allocations.jl") # Batch scaling + NamedTuple regression guards
    include("test_ad_chunk_mode.jl") # AD Jacobians with > 12 variables (ForwardDiff chunk mode)
    include("test_api_contracts.jl") # exported names defined; output-buffer size checked
    include("test_public_api.jl") # public API used by Margins (accessors, hooks, private scratch)
    include("test_unsupported_functions.jl") # unsupported formula functions fail at compile time
    include("test_threaded_level_cache.jl") # LevelIndex pool cache under concurrent evaluation
    include("test_compiled_cache.jl") # Bounded/clearable compiled-formula cache (LRU, release, concurrency)

    # Regression tests
    include("test_fd_jacobian_column.jl") # fd_jacobian_column! (incl. log(x) row-1 regression), FD at extreme scales
    # Edge-case regression and stability tests
    include("test_derivatives_domain_edge_cases.jl")
    include("test_link_derivatives.jl") # Link dμ/dη, d²μ/dη² vs BigFloat references (stability, Power/NegBin links)
    
    # External package integration
    include("test_standardized_predictors.jl") # StandardizedPredictors.jl integration
    include("test_centered_scaled_predictors.jl") # Center() and Scale() (CenteredTerm, ScaledTerm)
    include("test_glm_integration.jl") # GLM integration testing
    include("test_mixedmodels_integration.jl") # MixedModels integration testing
    include("test_mixedmodels_extension.jl") # MixedModels as a weak dependency (extension; FC without MixedModels)

    # Performance testing
    include("test_large_dataset_performance.jl") # Large dataset performance validation
end
