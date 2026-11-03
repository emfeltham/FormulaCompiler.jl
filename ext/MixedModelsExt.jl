"""
    MixedModelsExt

Package extension adding MixedModels.jl support to FormulaCompiler. It is loaded
automatically when MixedModels is loaded together with FormulaCompiler, so FormulaCompiler
no longer loads MixedModels for users who do not need it.

Adds:
- `FormulaCompiler.fixed_effects_form` methods for `LinearMixedModel` and
  `GeneralizedLinearMixedModel`, which drop every random-effects term (`(1|g)`,
  `(1 + x|g)`, `zerocorr(...)`, nested `(1|a/b)`), so `compile_formula`, `modelrow`, the
  derivative evaluators and `contrastevaluator` compile the fixed effects only;
- a `decompose_term!` method that skips any random-effects term that reaches the compiler.
"""
module MixedModelsExt

using FormulaCompiler
using FormulaCompiler: CompilationContext
using StatsModels
import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel

# Every random-effects term type (RandomEffectsTerm, ZeroCorr from `zerocorr(...)`, ...):
# the same hierarchy in MixedModels 4.34–5.x.
const RET = MixedModels.AbstractReTerm
# Unapplied `(… | …)` calls, should a formula carry one
const FT = StatsModels.FunctionTerm{typeof(|)}

"""
    fixed_effects_form(model::LinearMixedModel)
    fixed_effects_form(model::GeneralizedLinearMixedModel)

Remove any random-effects terms `( … | … )` from the RHS and return the
pure fixed-effects formula.
"""
function FormulaCompiler.fixed_effects_form(model::Union{LinearMixedModel,
                                                         GeneralizedLinearMixedModel})
    full = StatsModels.formula(model)   # e.g. y ~ x + z + x&z + (1|g)
    rhs  = full.rhs                     # tuple of top-level terms

    # drop every random-effects term and any unapplied FunctionTerm for `|`
    fe = filter(t -> !(t isa RET) && !(t isa FT), rhs)

    # if nothing was removed, just hand back the original
    fe === rhs && return full

    # otherwise rebuild: if you stripped *all* terms, leave only the intercept
    new_rhs = isempty(fe) ? StatsModels.ConstantTerm(1) : reduce(+, fe)
    return full.lhs ~ new_rhs
end

# Random effects are not part of the compiled (fixed-effects) model row.
function FormulaCompiler.decompose_term!(ctx::CompilationContext, term::RET, data_example)
    return Int[]
end

end # module
