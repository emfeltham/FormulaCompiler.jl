"""
    FormulaCompiler

High-performance, zero-allocation statistical formula evaluation for Julia.

## Two-Phase Compilation Architecture

FormulaCompiler uses a sophisticated two-phase compilation system to achieve maximum performance:

### Phase 1: Complete Compilation (CompiledFormula)
- **Function**: `compile_formula_complete(model, data)`
- **Result**: `CompiledFormula` - evaluator tree-based representation
- **Purpose**: Complete formula parsing, analysis, and validation
- **Performance**: Good (~100ns per row)
- **Use case**: When you need the intermediate representation or debugging

### Phase 2: Performance Optimization (SpecializedFormula)  
- **Function**: `compile_formula_optimized(compiled_formula)`
- **Result**: `SpecializedFormula` - tuple-based specialized representation
- **Purpose**: Maximum runtime performance through type specialization
- **Performance**: Exceptional (~50ns per row, zero allocations)
- **Use case**: Production code where speed is critical

### Main API
- **`compile_formula(model, data)`**: Complete + optimization in one call (recommended)

## Key Design Principles

1. **CompiledFormula is the foundation**: It handles all the complex parsing and creates 
   a complete, functional evaluator tree representation.

2. **SpecializedFormula is the optimization**: It analyzes the CompiledFormula structure 
   and creates specialized, type-stable execution paths.

3. **Both systems are fully functional**: You can execute either representation, 
   but SpecializedFormula provides superior performance.

## Example Usage

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Your data
df = DataFrame(x = randn(1000), group = rand(["A", "B"], 1000))
df.y = df.x + randn(1000)
model = lm(@formula(y ~ x * group), df)
data = Tables.columntable(df)

# Option 1: Two-phase compilation (for when you need intermediate form)
compiled = compile_formula_complete(model, data)  # CompiledFormula
specialized = compile_formula(compiled)           # SpecializedFormula

# Option 2: Direct compilation (recommended for most use cases)
formula = compile_formula(model, data)            # SpecializedFormula directly

# High-performance execution (zero allocations)
output = Vector{Float64}(undef, length(formula))
for i in 1:nrow(df)
    formula(output, data, i)  # ~50ns, 0 allocations
end
```
"""
module FormulaCompiler

################################ Dependencies ################################

# Development dependencies (remove from production builds)
using Random, Test, BenchmarkTools

# Core dependencies
using Dates: now
using Statistics
using StatsModels, GLM, CategoricalArrays, Tables, DataFrames
using LinearAlgebra: dot, I
using ForwardDiff
using Base.Iterators: product # -> compute_kronecker_pattern

# External package integration
import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel
using StandardizedPredictors: ZScoredTerm

################################# Core System #################################

# Core utilities and types
include("core/utilities.jl")
export not, OverrideVector

################################# Integration #################################

# External package integration
include("integration/mixed_models.jl")

################################# Compilation #################################

# Compilation system (unified)
include("compilation/compilation.jl")

export compile_formula, compile_unified

################################## Scenarios ##################################

# Override and scenario system (needed by modelrow)
include("scenarios/overrides.jl")
export create_categorical_override, create_scenario_grid
export DataScenario, create_scenario, create_override_data, create_override_vector

################################# Evaluation #################################

# High-level evaluation interface
include("evaluation/modelrow.jl")
export ModelRowEvaluator, modelrow!, modelrow

############################## Development Tools ##############################

# Development utilities (only include in dev builds)
include("dev/testing_utilities.jl")
export test_correctness, test_data, make_test_data, test_model_correctness

############################## Future Features ##############################

# Derivative system (under development)
# include("derivatives/step1_foundation.jl")
# include("derivatives/step2_functions.jl")
# export compile_derivative_formula

end # end module