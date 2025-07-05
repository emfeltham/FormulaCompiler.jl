module EfficientModelMatrices

using StatsModels
using Tables
using DataFrames: nrow
using CategoricalArrays

include("modelmatrix!.jl")
include("helpers.jl")

# From modelmatrix!.jl
export modelmatrix!

# From matrix_reuse_helpers.jl  
export extract_model_matrix
export analyze_variable_dependencies_fast
export estimate_term_width_fast
export find_variables_in_term_fast

end # module EfficientModelMatrices
