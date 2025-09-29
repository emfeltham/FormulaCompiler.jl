# contrasts.jl - Discrete contrasts for categorical variables

# NOTE: Basic contrast_modelrow! function has been removed.
# Use ContrastEvaluator for efficient zero-allocation categorical contrasts:
#
# Example:
#   evaluator = contrastevaluator(compiled, data, [:var])
#   contrast_buf = Vector{Float64}(undef, length(compiled))
#   contrast_modelrow!(contrast_buf, evaluator, row, :var, from, to)
#
# This provides zero-allocation performance and is the recommended API.