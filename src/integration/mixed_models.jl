###############################################################################
# fixed_effects_form(model)  – return *only* the fixed part of the formula
#
# The generic function and the methods for models without random effects live here.
# The MixedModels methods (LinearMixedModel, GeneralizedLinearMixedModel, which drop
# every `MixedModels.AbstractReTerm`) are added by the package extension
# `ext/MixedModelsExt.jl` when MixedModels is loaded. Other packages may add methods
# (Margins does for FixedEffectModels).
###############################################################################

"""
    fixed_effects_form(model) -> FormulaTerm

Return the schema-applied formula of the fixed-effects part of `model`. `compile_formula`
compiles this formula.

For models without random effects (GLM `lm`/`glm` fits, `TableRegressionModel`s and, by
default, any other StatsModels model) this is `StatsModels.formula(model)`. For
`MixedModels.LinearMixedModel` and `GeneralizedLinearMixedModel` (methods defined in the
MixedModels package extension) every random-effects term `( … | … )`, including
`zerocorr(...)` terms, is removed.

# Model-class hook

This is the public extension point for model types whose `StatsModels.formula` is not the
formula of the model matrix ``X`` (for example, one that also carries absorbed fixed effects
or other terms with no column in ``X``). A package supporting such a model type adds a method:

```julia
import FormulaCompiler
FormulaCompiler.fixed_effects_form(m::MyModel) = my_fixed_part(m)  # schema-applied FormulaTerm
```

The returned formula's right-hand side must produce exactly the columns of `coef(model)`,
in order. Margins does this for `FixedEffectModels.FixedEffectModel`.
"""
function fixed_effects_form end

fixed_effects_form(model) = StatsModels.formula(model)

fixed_effects_form(model::StatsModels.TableRegressionModel) = formula(model)

fixed_effects_form(model::Union{LinearModel, GeneralizedLinearModel}) = formula(model)

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

function extract_columns_recursive!(columns::Vector{Symbol}, term::CenteredTerm)
    extract_columns_recursive!(columns, term.term)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::ScaledTerm)
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
