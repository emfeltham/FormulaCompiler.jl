# Load your existing packages
using Margins, EfficientModelMatrices, StatsModels, GLM, DataFrames, Tables

# Load ONLY the recursive implementation
include("phase1_complete_recursion.jl")   # Complete recursive system
include("phase1_testing.jl")              # Updated tests
include("phase1_validation_suite.jl")     # Updated validation  
include("phase1_test_runner.jl")          # Updated runner

using CategoricalArrays

#
include("categorical_debug_tools.jl")

# Test 1: Categorical term alone (should be perfect now)
debug_categorical_term_standalone()

# Test 2: The original failing interaction  
test_simple_interaction()

# Test 3: Step-by-step interaction debugging
debug_interaction_step_by_step()

# Run validation
validate_phase1()

##

include("phase1_performance_fix.jl")
validate_phase1_fixed()