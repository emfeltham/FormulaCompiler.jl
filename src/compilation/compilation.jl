# UnifiedCompiler Main Compilation
# Entry point for unified formula compilation

using StatsModels
using GLM
using MixedModels
using StandardizedPredictors

include("types.jl")
include("execution.jl")
include("scratch.jl")
include("decomposition.jl")

# Helper to extract fixed effects formula
function get_fixed_effects_formula(model)
    # For MixedModels, extract only fixed effects
    if isdefined(MixedModels, :LinearMixedModel) && isa(model, MixedModels.LinearMixedModel)
        return fixed_effects_form(model)
    elseif isdefined(MixedModels, :GeneralizedLinearMixedModel) && isa(model, MixedModels.GeneralizedLinearMixedModel)
        return fixed_effects_form(model)
    else
        # For GLM and other models, use the full formula
        return StatsModels.formula(model)
    end
end

# Import the fixed_effects_form function
include("../integration/mixed_models.jl")

# Primary API: Compile from model (has schema-applied formula)
function compile_formula(model, data_example::NamedTuple)
    # Extract schema-applied formula using standard API
    # For MixedModels, this extracts only fixed effects
    formula = get_fixed_effects_formula(model)
    
    # Decompose formula to operations (formula has schema info)
    ops_vec, scratch_size, output_size = decompose_formula(formula, data_example)
    
    # TODO: Dependency resolution
    # ops_vec = resolve_dependencies(ops_vec)
    
    # TODO: Optimization passes
    # ops_vec = optimize_operations(ops_vec)
    
    # Convert to tuple for type stability
    ops_tuple = Tuple(ops_vec)
    
    # Create specialized compiled formula
    return UnifiedCompiled{typeof(ops_tuple), scratch_size, output_size}(ops_tuple)
end

# Secondary API: Direct formula compilation (for testing)
function compile_formula(formula::FormulaTerm, data_example::NamedTuple)
    # Warning: This formula may not have schema applied
    # Better to create a model first for proper schema application
    ops_vec, scratch_size, output_size = decompose_formula(formula, data_example)
    ops_tuple = Tuple(ops_vec)
    return UnifiedCompiled{typeof(ops_tuple), scratch_size, output_size}(ops_tuple)
end

# Alternative entry point that matches current API
function compile_unified(model, data::NamedTuple)
    return compile_formula(model, data)
end

# Export main functions
export UnifiedCompiled, compile_formula, compile_unified