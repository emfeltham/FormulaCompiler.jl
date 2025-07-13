# usage.jl
# Complete Usage Example

using Revise
using BenchmarkTools
using EfficientModelMatrices

using DataFrames, GLM, Tables, CategoricalArrays
using CategoricalArrays: CategoricalValue, levelcode
using StatsModels
using StandardizedPredictors: ZScoredTerm
using Random


# 1. Create your data and model
df = DataFrame(
    x = randn(1000),
    y = randn(1000),
    z = abs.(randn(1000)) .+ 0.1,
    group = categorical(rand(["A", "B"], 1000))
);

model = lm(@formula(y ~ x + x^2 * log(z) + group), df)
data = Tables.columntable(df)

# 2. One-time compilation (expensive, ~1-10ms)
fc = compile_formula(model)

# 3. Setup for fast evaluation
i = 
row_vec = Vector{Float64}(undef, fc.output_width);
modelrow!(row_vec, fc.formula_val, data, i)
@assert modelmatrix(model)[i, :] == row_vec;

# 4. Zero-allocation runtime usage (~50-100ns per call)
function rowloop!(row_vec, fc, data)
    for i in 1:1000
        modelrow!(row_vec, fc.formula_val, data, i)
        # Now row_vec contains the model matrix row for observation i
        # Use row_vec for predictions, marginal effects, etc.
    end
end

fill!(row_vec, 0.0);
@btime rowloop!(row_vec, fc, data);

# 5. Performance testing
test_compilation_performance(model, data)
```

# Alternative: Direct function access (even faster)

```julia
# Get direct function reference
func, output_width, column_names = get_compiled_function(model)

# Use directly (bypasses @generated dispatch)
func(row_vec, data, 1)  # Potentially 10-20% faster
