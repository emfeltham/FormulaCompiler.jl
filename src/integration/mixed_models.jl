###############################################################################
# fixed_helpers.jl
#
# fixed_effects_form(model)  – return *only* the fixed part of the formula
#
# These functions are agnostic to whether `model` came from GLM.jl or
# MixedModels.jl; they always reuse the exact same contrasts / dummy coding /
# intercept handling that the fitted model stored in its schema.
# Dispatch on model type so GLM/OLS do nothing, MixedModels get stripped.
###############################################################################

# also need these to manipulate the formula
const RET = MixedModels.RandomEffectsTerm
const FT  = StatsModels.FunctionTerm{typeof(|)}
const CT  = StatsModels.ConstantTerm

# ─────────────────────────────────────────────────────────────────────────────
# 1) GLM/OLS methods: identity
# ─────────────────────────────────────────────────────────────────────────────

fixed_effects_form(model::StatsModels.TableRegressionModel) = formula(model)

"""
    fixed_effects_form(model::LinearModel)
    fixed_effects_form(model::GeneralizedLinearModel)

For plain OLS (`lm`) or GLM (`glm`) fits, there are no random-effects terms, 
so we just return the original formula unchanged.
"""
fixed_effects_form(model::Union{LinearModel, GeneralizedLinearModel}) = formula(model)

# ─────────────────────────────────────────────────────────────────────────────
# 2) MixedModels methods: strip out `(…|…)`
# ─────────────────────────────────────────────────────────────────────────────

"""
    fixed_effects_form(model::LinearMixedModel)
    fixed_effects_form(model::GeneralizedLinearMixedModel)

Remove any random-effects terms `( … | … )` from the RHS and return the
pure fixed-effects formula.
"""
function fixed_effects_form(model::Union{LinearMixedModel,
                                          GeneralizedLinearMixedModel})
    full = formula(model)      # e.g. y ~ x + z + x&z + (1|g)
    rhs  = full.rhs            # vector of top‐level terms

    # drop any RandomEffectsTerm or the FunctionTerm for `|`
    fe = filter(t -> !(t isa RET) && !(t isa FT), rhs)

    # if nothing was removed, just hand back the original
    fe === rhs && return full

    # otherwise rebuild: if you stripped *all* terms, leave only the intercept
    new_rhs = isempty(fe) ? CT() : reduce(+, fe)
    return full.lhs ~ new_rhs
end

###############################################################################
# 8. EXTRACTION UTILITIES
###############################################################################

function extract_all_columns(term::AbstractTerm)
    columns = Symbol[]
    extract_columns_recursive!(columns, term)
    return unique(columns)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::Union{ContinuousTerm, Term})
    push!(columns, term.sym)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::CategoricalTerm)
    push!(columns, term.sym)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::FunctionTerm)
    for arg in term.args
        extract_columns_recursive!(columns, arg)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::InteractionTerm)
    for comp in term.terms
        extract_columns_recursive!(columns, comp)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::ZScoredTerm)
    extract_columns_recursive!(columns, term.term)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::MatrixTerm)
    for sub_term in term.terms
        extract_columns_recursive!(columns, sub_term)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::Union{InterceptTerm, ConstantTerm})
    # No columns
end
