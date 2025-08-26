# extract_example.jl
# demonstrate how information is stored in categorical term

using Revise
using Test
using BenchmarkTools, Profile
using FormulaCompiler
using Random

using Statistics
using DataFrames, Tables, CategoricalArrays
using StatsModels, StandardizedPredictors
using MixedModels, GLM

using FormulaCompiler: make_test_data

fx = @formula(continuous_response ~ x + group3 + (1|subject)),
model = fit(MixedModel, fx, df; progress = false)
fe = FormulaCompiler.fixed_effects_form(model)
rhs = fe.rhs

rhs.terms[end]
t = rhs.terms[end]
c = t.contrasts
c.contrasts
c.coefnames
c.matrix
