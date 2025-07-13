# _cols workflow.jl

using EfficientModelMatrices
using Profile
using BenchmarkTools

import EfficientModelMatrices._cols!

# Recursive unrolling (more elegant)
@inline function _cols!(row_vec, terms::Tuple, data, pos=1)
    return _cols_unroll!(row_vec, terms, data, pos)
end

@inline _cols_unroll!(row_vec, ::Tuple{}, data, pos) = pos

@inline function _cols_unroll!(row_vec, terms::Tuple, data, pos)
    new_pos = _cols!(row_vec, first(terms), data, pos)
    return _cols_unroll!(row_vec, Base.tail(terms), data, new_pos)
end

using Margins: fixed_effects_form
#using EfficientModelMatrices:  _get_or_build_cache

form5 = @formula(SepalLength ~ SepalWidth * Species  + SepalWidth^2)
form5 = @formula(SepalLength ~ SepalWidth^2)
m5 = lm(form5, iris)
levels_list = levels(iris.Species)
repvals_cat = Dict(:Species => levels_list)

model = m5;
data = columntable(iris);

rhs = fixed_effects_form(model).rhs;

row_vec = fill(0.0, size(modelmatrix(model), 2));

@btime _cols!(row_vec, rhs, data)

# @profile [_cols!(row_vec, rhs, data) for _ in 1:10];


@time rhs_c = collect(rhs.terms);

row_vec .= 0.0
@btime _cols!(row_vec, rhs_c, data);
@code_warntype _cols!(row_vec, rhs_c, data);

##

using Margins: build_cols_cache

@btime cache = build_cols_cache(rhs.terms)  # Expensive but only once

@btime _cols_cached!(row_vec, rhs.terms, data)  # Uses pre-compiled evaluators

@allocated cache = _get_or_build_cache(rhs.terms)

using EfficientModelMatrices: COLS_CACHES, _cols_with_cache!

cache2 = COLS_CACHES[hash(rhs.terms)]
@allocated _cols_with_cache!(row_vec, rhs.terms[1], data, 1, cache)

println("Hash lookup allocations:")
term = rhs.terms[1]
@allocated cache.term_hashes[term]

###

# No lookups, no conditionals, just direct function calls:
for i in 1:length(evaluators)
    row_vec[positions[i]] = evaluators[i](data, row_index)
end

# Usage Pattern:
# One-time setup (expensive, but amortized)
evaluator = create_precomputed_evaluator(model)
row_vec = Vector{Float64}(undef, evaluator.total_width)

row_index = 1

# Ultra-fast evaluation (use in AME loops)
@btime ultra_fast_modelrow!(row_vec, evaluator, data, row_index)  # ~80ns, 0 alloc

@profile [ultra_fast_modelrow!(row_vec, evaluator, data, row_index) for _ in 1:10]
open("profile_output.txt", "w") do io
    Profile.print(io)
end

# Works with overrides for MERs
rep_data = create_override_data(data, Dict(:x => 2.5));
@btime ultra_fast_modelrow!(row_vec, evaluator, rep_data, row_index)  # Same performance


###

@btime evaluator.evaluators[1](data, 1);
@btime evaluator.evaluators[2](data, 1);

###

function test_direct_data_access(data, v, row_index)
    @inbounds val = Float64(data[v][row_index])
    return val
end

v = :PetalWidth
@btime test_direct_data_access($data, $v, 1)  # Should be 0 allocations

###

function test_direct_power(data, v, row_index)
    @inbounds val = Float64(data[v][row_index])
    return val * val
end

@btime test_direct_power($data, $v, 1)  # Should be 0 allocations

###

# Manual construction (should work):
function manual_continuous_evaluator(data, row_index)
    @inbounds return Float64(data.PetalWidth[row_index])
end

@btime manual_continuous_evaluator($data, 1)  # Should be 0 allocations

# vs.

# Closure construction (probably allocates):
sym = :PetalWidth
closure_evaluator = (data, row_index) -> Float64(data[sym][row_index])

@btime closure_evaluator($data, 1)  # Probably allocates

###

@btime predict(model);
@btime predict(model, iris[1:1, :])
@btime predict(model, iris[1:10, :]);
@btime predict(model, iris[1:end, :]);

@btime predict2(model, iris[1:end, :]);