# compiled_formula_testing.jl

using Revise
using BenchmarkTools
using EfficientModelMatrices
using EfficientModelMatrices: fixed_effects_form

using StatsModels
import CategoricalArrays
using CategoricalArrays: CategoricalValue, levelcode

begin
    using Test
    using Random
    using DataFrames, CategoricalArrays
    using Distributions, Statistics, GLM, MixedModels
    # using RDatasets
    import LinearAlgebra.dot
    import LinearAlgebra.diag
    using StandardizedPredictors
end

using EfficientModelMatrices:
    compile_formula,
    get_direct_function,
    generate_loop_function,
    # generated
    compile_formula_generated,
    FORMULA_CACHE,
    register_formula!,
    modelrow!

# Simulate data
n = 50_000;
large_df = DataFrame(
    x = randn(n), 
    y = randn(n), 
    z = abs.(randn(n)) .+ 1,
    group = categorical(rand(["A", "B", "C"], n))
);
data = Tables.columntable(large_df);

model = lm(@formula(y ~ x + x^2 + log(z) + group), large_df);
mm = modelmatrix(model);

compiled = compile_formula(model);
direct_func = get_direct_function(compiled);

# compiled.output_width
# this is probably the number of levels too!

# Test full matrix evaluation time
row_vec = Vector{Float64}(undef, size(mm, 2));
fill!(row_vec, 0.0);

# this really is no allocations
@btime direct_func(row_vec, data, 1);  # Target: ~10ns, 0 allocations

mm1 = vec(mm[1, :])
row_vec

hcat(mm1, row_vec)

@assert isapprox(mm1, row_vec)

###### compiled eval loop

loop_func = generate_loop_function(compiled, n)
fill!(row_vec, 0.0)
@btime loop_func(row_vec, data, n)  # Should be 0 allocations

mm = modelmatrix(model);

@assert row_vec == mm[n, :];

### generated

# Test the @generated approach
formula_val, output_width, column_names = compile_formula_generated(model)
row_vec_gen = Vector{Float64}(undef, output_width);

# Single call test
@btime modelrow!(row_vec_gen, formula_val, data, 1)

# Verify correctness  
mm_row = modelmatrix(model)[1, :];
println("Correct:   ", mm_row)
println("Generated: ", row_vec_gen)
println("Match: ", isapprox(mm_row, row_vec_gen))

# following generated method:
# normal loop construction works, and doesn't allocate
function modelrows_1!(row_vecs, formula_val, data)
    for i in eachindex(row_vecs)
        modelrow!(row_vecs[i], formula_val, data, i)
    end
end

row_vecs = [fill(0.0, length(row_vec_gen)) for _ in 1:1000];
@btime modelrows_1!(row_vecs, formula_val, data);

@assert permutedims(reduce(hcat, row_vecs)) == mm[1:1000, :]

#

# I am not sure what this is about:
# Loop test - can use your existing loop generation but with the @generated function
# function generate_loop_function_generated(formula_val, n)
#     loop_name = Symbol("loop_generated_$(typeof(formula_val).parameters[1])")
    
#     loop_code = """
#     function $loop_name(row_vec, formula_val, data, n)
#         @inbounds for i in 1:n
#             ultra_fast_modelrow!(row_vec, formula_val, data, i)
#         end
#     end
#     """
    
#     eval(Meta.parse(loop_code))
#     return getproperty(Main, loop_name)
# end

# loop_func_gen = generate_loop_function_generated(formula_val, n)
# @btime loop_func_gen(row_vec_gen, formula_val, data, n)

## standardization

contrasts = Dict(:x => ZScore());
model = lm(@formula(y ~ x + x^2 + log(z) + group), large_df; contrasts);

formula_val, output_width, column_names = compile_formula_generated(model)
row_vec_gen = Vector{Float64}(undef, output_width);

# Single call test
@btime modelrow!(row_vec_gen, formula_val, data, 1)

@assert row_vec_gen == modelmatrix(model)[1, :];

#####

contrasts = Dict(:x => ZScore());
model = lm(@formula(y ~ x + x^2 * log(z) + group), large_df; contrasts);

formula_val, output_width, column_names = compile_formula_generated(model)
row_vec_gen = Vector{Float64}(undef, output_width);

# Single call test
@btime modelrow!(row_vec_gen, formula_val, data, 1)

@assert row_vec_gen == modelmatrix(model)[1, :];

