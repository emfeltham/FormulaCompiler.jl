# test_zero_allocation_overrides.jl - Zero allocation validation for Part 0 override improvements
#
# Validates that modelrow! with scenarios achieves zero allocations after warmup
# per MARGINS_COMPUTE_REWRITE_PLAN.md Part 0 requirements

using Test
using FormulaCompiler
using GLM, DataFrames, Tables
using BenchmarkTools
using CategoricalArrays
using FormulaCompiler: NumericCounterfactualVector, BoolCounterfactualVector, CategoricalCounterfactualVector, modelrow!

# Function barrier: warm up, then measure one modelrow! call
function _override_modelrow_bytes(output, compiled, data, row)
    modelrow!(output, compiled, data, row)
    return @allocated modelrow!(output, compiled, data, row)
end

@testset "Zero Allocation Override Performance" begin

    # Setup test data
    n = 100
    df = DataFrame(
        y = randn(n),
        x1 = randn(n),
        x2 = randn(n),
        age = rand(18:65, n),
        treated = rand(Bool, n),
        group = categorical(rand(["A", "B", "C"], n), levels=["A", "B", "C"]),
        region = categorical(rand(["North", "South"], n), levels=["North", "South"])
    )

    data_nt = Tables.columntable(df)
    model = lm(@formula(y ~ x1 + x2 + treated + group), df)
    compiled = compile_formula(model, data_nt)

    # Complex model with interactions and transforms
    complex_df = DataFrame(
        y = randn(50),
        x1 = randn(50),
        x2 = randn(50),
        age = rand(25:60, 50),
        treated = rand(Bool, 50),
        group = categorical(rand(["A", "B"], 50), levels=["A", "B"])
    )
    complex_data = Tables.columntable(complex_df)
    complex_model = lm(@formula(y ~ x1 + log(age) + treated * x2 + group + x1^2), complex_df)
    complex_compiled = compile_formula(complex_model, complex_data)

    # (name, compiled formula, data with counterfactual columns)
    cases = [
        ("Continuous Variable Overrides", compiled,
            merge(data_nt, (x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 2.0),
                            x2 = NumericCounterfactualVector{Float64}(data_nt.x2, 1, -1.5)))),
        ("Boolean Variable Overrides", compiled,
            merge(data_nt, (treated = BoolCounterfactualVector(data_nt.treated, 1, true),))),
        ("Categorical Variable Overrides", compiled,
            merge(data_nt, (group = CategoricalCounterfactualVector(data_nt.group, 1, CategoricalValue("B", data_nt.group)),
                            region = CategoricalCounterfactualVector(data_nt.region, 1, CategoricalValue("South", data_nt.region))))),
        ("Mixed Variable Overrides", compiled,
            merge(data_nt, (x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 1.5),
                            x2 = NumericCounterfactualVector{Float64}(data_nt.x2, 1, -0.8),
                            age = NumericCounterfactualVector{Int64}(data_nt.age, 1, 40),
                            treated = BoolCounterfactualVector(data_nt.treated, 1, false),
                            group = CategoricalCounterfactualVector(data_nt.group, 1, CategoricalValue("C", data_nt.group)),
                            region = CategoricalCounterfactualVector(data_nt.region, 1, CategoricalValue("North", data_nt.region))))),
        ("Normal vs Override: x1 and treated", compiled,
            merge(data_nt, (x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 2.0),
                            treated = BoolCounterfactualVector(data_nt.treated, 1, true)))),
        ("Complex Model Zero Allocation", complex_compiled,
            merge(complex_data, (x1 = NumericCounterfactualVector{Float64}(complex_data.x1, 1, 1.5),
                                 age = NumericCounterfactualVector{Int64}(complex_data.age, 1, 35),
                                 treated = BoolCounterfactualVector(complex_data.treated, 1, true),
                                 x2 = NumericCounterfactualVector{Float64}(complex_data.x2, 1, -0.5),
                                 group = CategoricalCounterfactualVector(complex_data.group, 1, CategoricalValue("B", complex_data.group))))),
    ]

    for (name, comp, cf_data) in cases
        @testset "$name" begin
            output = Vector{Float64}(undef, length(comp))
            @test _override_modelrow_bytes(output, comp, cf_data, 1) == 0
            @test all(isfinite, output)
        end
    end

    @testset "Multiple Rows Zero Allocation" begin
        cf_x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 2.5)
        cf_treated = BoolCounterfactualVector(data_nt.treated, 1, true)
        group_a = CategoricalValue("A", data_nt.group)
        cf_group = CategoricalCounterfactualVector(data_nt.group, 1, group_a)
        cf_data = merge(data_nt, (x1 = cf_x1, treated = cf_treated, group = cf_group))
        output = Vector{Float64}(undef, length(compiled))

        for row_idx in [1, 10, 25, 50, 75, 100]
            # Update counterfactual to apply to this row
            FormulaCompiler.update_counterfactual_row!(cf_x1, row_idx)
            FormulaCompiler.update_counterfactual_row!(cf_treated, row_idx)
            FormulaCompiler.update_counterfactual_row!(cf_group, row_idx)

            @test _override_modelrow_bytes(output, compiled, cf_data, row_idx) == 0
        end
    end

    @testset "Comparison: Normal vs Override Performance" begin
        # Allocations: the plain row is checked in test_allocations.jl and
        # test_formulacompiler_primitives_allocations.jl (modelrow!, Bool predictor);
        # this override row in the "Normal vs Override" case above.
        output = Vector{Float64}(undef, length(compiled))
        cf_data = merge(data_nt, (x1 = NumericCounterfactualVector{Float64}(data_nt.x1, 1, 2.0),
                                  treated = BoolCounterfactualVector(data_nt.treated, 1, true)))

        modelrow!(output, compiled, data_nt, 1)  # Warmup
        modelrow!(output, compiled, cf_data, 1)  # Warmup
        normal_result = @benchmark modelrow!($output, $compiled, $data_nt, 1) samples=200 seconds=2
        override_result = @benchmark modelrow!($output, $compiled, $cf_data, 1) samples=200 seconds=2

        # Performance should be comparable (within 2x)
        performance_ratio = minimum(override_result.times) / minimum(normal_result.times)
        @test performance_ratio < 2.0
    end
end
