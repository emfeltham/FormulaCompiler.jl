# comp.jl

using Revise
using BenchmarkTools
using EfficientModelMatrices: fixed_effects_form
using DataFrames, GLM, CategoricalArrays
using Random


###
using StandardizedPredictors
using StandardizedPredictors: ZScore, ZScoredTerm

include("phase1_structure_analysis.jl")
include("phase2_instruction_generation.jl")
include("phase3_generated_integration.jl")

##

test_structure_analysis()

test_structure_analysis_standard()


test_instruction_generation()


test_complete_pipeline()

######## DEBUG

Random.seed!(42)
n = 100
df = DataFrame(
    x = randn(n),
    y = randn(n),
    z = abs.(randn(n)) .+ 0.1,  # Positive for log
    w = randn(n),
    group = categorical(rand(["A", "B", "C"], n)),
    binary = categorical(rand(["Yes", "No"], n))
)

data = Tables.columntable(df);

model = lm(@formula(y ~ 1), data);
mm = modelmatrix(model);
row_vec = similar(mm, 1)

# Manual function for formula hash 10335344282077925378 (intercept only)
function manual_intercept_only(row_vec, data, row_idx)
    @inbounds row_vec[1] = 1.0
    return row_vec
end

# Test this directly
@allocated manual_intercept_only(row_vec, data, 1)  # Should be 0

# test 2
formula_val, output_width, column_names = compile_formula_complete(model)

@code_typed modelrow!(row_vec, formula_val, data, 1)
@code_llvm modelrow!(row_vec, formula_val, data, 1)


@btime modelrow!(row_vec, formula_val, data, 1)

###

test_instruction_generation()
test_interaction_fix()


using CategoricalArrays
using StatsModels

# Check what the actual type looks like
comp_type = CategoricalTerm{DummyCoding, Matrix{Float64}, 1}
println("comp_type: $comp_type")
println("comp_type <: CategoricalTerm: $(comp_type <: CategoricalTerm)")

### re test pipeline

test_complete_pipeline()