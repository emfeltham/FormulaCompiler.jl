# allocation_survey.jl

# julia --project="." test/allocation_survey.jl > allocation_survey_output.txt 2>&1

# Comprehensive allocation survey - clean version
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, MixedModels
using StatsModels, BenchmarkTools, CSV
using FormulaCompiler: make_test_data, test_formulas
using Random

include("../src/unified/compilation.jl")

Random.seed!(08540)

# Setup
n = 500
df = make_test_data(; n);
data = Tables.columntable(df);

results_df = DataFrame(
    category = String[],
    name = String[],
    model_type = String[],
    model_size = Int[],
    memory_bytes = Int[],
    time_ns = Float64[],
    status = String[]
);

# Benchmark function with proper warmup and measurement
function benchmark_model!(results_df, category, name, model, model_type)
    compiled = compile_formula_unified(model, data)
    buffer = Vector{Float64}(undef, length(compiled))
    
    # Extensive warmup to ensure compilation is complete
    for i in 1:100
        compiled(buffer, data, 1)
    end
    
    # Benchmark with proper settings for accurate allocation measurement
    benchmark_result = @benchmark $compiled($buffer, $data, 1) samples=1000 seconds=2
    
    memory_bytes = minimum(benchmark_result.memory)
    time_ns = minimum(benchmark_result.times)
    
    status = if memory_bytes == 0
        "✅ PASS"
    elseif memory_bytes < 10
        "❌ FAIL"
    else
        "❌ BIG FAIL"
    end
    
    push!(results_df, (
        category = category,
        name = name,
        model_type = model_type,
        model_size = length(compiled),
        memory_bytes = memory_bytes,
        time_ns = time_ns,
        status = status
    ))
end

## example - proper measurement without artifacts
function example_run!(compiled, data)
    buffer = Vector{Float64}(undef, length(compiled))
    
    # Extensive warmup to eliminate compilation overhead
    for i in 1:100
        compiled(buffer, data, 1)
    end
    
    # Use @benchmark for accurate allocation measurement
    return @benchmark $compiled($buffer, $data, 1) samples=1000 seconds=2
end;

i = 2
fx = test_formulas.lm[i]
model = lm(fx.formula, df)
compiled = compile_formula_unified(model, data)
# example_run!(compiled, data)

buffer = Vector{Float64}(undef, length(compiled))
@btime compiled($buffer, $data, 1);
##

for fx in test_formulas.lm
    model = lm(fx.formula, df)
    benchmark_model!(results_df, "LM", fx.name, model, "LinearModel")
end

for fx in test_formulas.glm
    model = glm(fx.formula, df, fx.distribution, fx.link)
    benchmark_model!(results_df, "GLM", fx.name, model, "GeneralizedLinearModel")
end

for fx in test_formulas.lmm
    model = fit(MixedModel, fx.formula, df; progress = false)
    benchmark_model!(results_df, "LMM", fx.name, model, "LinearMixedModel")
end

for fx in test_formulas.glmm
    model = fit(MixedModel, fx.formula, df, fx.distribution, fx.link; progress = false)
    benchmark_model!(results_df, "GLMM", fx.name, model, "GeneralizedLinearMixedModel")
end

# CHECK RESULTS

results_df
CSV.write("test/allocation_results.csv", results_df);
println("WRITTEN: test/allocation_results.csv")

show(results_df; allrows= true)
