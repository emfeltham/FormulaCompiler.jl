module EfficientModelMatrices

using StatsModels, Tables, DataFrames
using LinearAlgebra, SparseArrays
using StatsModels: schema, apply_schema, modelcols
using StatsModels: AbstractTerm, MatrixTerm
using CategoricalArrays

# Import MixedModels types if available
try
    using MixedModels
    using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel, RandomEffectsTerm
    const MIXEDMODELS_AVAILABLE = true
catch
    const MIXEDMODELS_AVAILABLE = false
    # Define dummy types to avoid errors
    abstract type LinearMixedModel end
    abstract type GeneralizedLinearMixedModel end
    abstract type RandomEffectsTerm end
end

include("structs.jl")
include("constructors.jl")
include("dependency_analysis.jl")
include("updaters.jl")
include("demo_usage.jl")

export
    CachedModelMatrix, cached_modelmatrix, update!, 
    TermDependencyCache, build_dependency_cache,
    selective_update!, batch_update!

end # module EfficientModelMatrices
