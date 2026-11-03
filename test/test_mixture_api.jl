# test_mixture_api.jl
# The user-facing mixture API: mix / CategoricalMixture (construction, validation, display,
# iteration), validate_mixture_against_data and mixture_to_scenario_value.
# Moved from Margins' test/features/test_categorical_mixtures.jl (these are FormulaCompiler's
# functions; Margins imports them). Mixture evaluation in compiled formulas is in
# test_mixture_evaluation.jl and test_mixture_modelrows.jl; validate_mixture_weights/_levels
# and two mix() errors are in test_api_contracts.jl.

using Test
using CategoricalArrays
using FormulaCompiler

@testset "Mixture API" begin
    @testset "mix and CategoricalMixture: construction" begin
        m1 = mix("A" => 0.3, "B" => 0.7)
        @test m1 isa CategoricalMixture{String}
        @test m1.levels == ["A", "B"]
        @test m1.weights == [0.3, 0.7]

        m2 = mix(:urban => 0.6, :rural => 0.4)
        @test m2 isa CategoricalMixture{Symbol}
        @test m2.levels == [:urban, :rural] && m2.weights == [0.6, 0.4]

        m3 = mix(true => 0.8, false => 0.2)
        @test m3 isa CategoricalMixture{Bool}
        @test m3.levels == [true, false] && m3.weights == [0.8, 0.2]

        # the levels keep the order given; integer weights become Float64
        m4 = mix("low" => 0.2, "medium" => 0.5, "high" => 0.3)
        @test m4.levels == ["low", "medium", "high"] && m4.weights == [0.2, 0.5, 0.3]
        m5 = mix("A" => 1, "B" => 0)
        @test m5.weights isa Vector{Float64} && m5.weights == [1.0, 0.0]

        m6 = CategoricalMixture(["X", "Y"], [0.4, 0.6])
        @test m6.levels == ["X", "Y"] && m6.weights == [0.4, 0.6]
    end

    @testset "mix and CategoricalMixture: invalid input" begin
        @test_throws ArgumentError mix("A" => 0.1, "B" => 0.1)          # sum below 1
        @test_throws ArgumentError mix("A" => 0.5, "B" => 0.5 + 1e-9)   # outside the 1e-10 tolerance
        @test_throws ArgumentError mix("A" => -0.1, "B" => 1.1)         # negative weight
        @test_throws ArgumentError mix("A" => 0.5, "B" => -0.5)
        @test_throws ArgumentError CategoricalMixture(["X", "X"], [0.5, 0.5])   # duplicate level
        @test_throws ArgumentError CategoricalMixture(["A", "B"], [0.3])        # length mismatch
        @test_throws ArgumentError CategoricalMixture(["A"], [0.3, 0.7])
        @test_throws ArgumentError mix()
    end

    @testset "display, length, indexing, iteration" begin
        m = mix("high" => 0.3, "low" => 0.7)
        @test string(m) == "mix(\"high\" => 0.3, \"low\" => 0.7)"
        @test string(mix(:a => 0.5, :b => 0.5)) == "mix(:a => 0.5, :b => 0.5)"
        @test string(mix(true => 0.8, false => 0.2)) == "mix(true => 0.8, false => 0.2)"
        @test length(m) == 2
        @test m[1] == ("high", 0.3)
        @test m[2] == ("low", 0.7)
        @test collect(m) == [("high", 0.3), ("low", 0.7)]
    end

    @testset "validate_mixture_against_data" begin
        # CategoricalArray: its levels, including unused ones
        edu = categorical(["high_school", "college", "graduate"])
        @test validate_mixture_against_data(mix("high_school" => 0.4, "college" => 0.6), edu, :education) === true
        @test validate_mixture_against_data(mix("high_school" => 0.3, "college" => 0.4, "graduate" => 0.3), edu, :education) === true
        err = try validate_mixture_against_data(mix("high_school" => 0.4, "phd" => 0.6), edu, :education); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("Variable :education mixture contains levels not found in data: [\"phd\"]", err.msg)
        unused = categorical(["a", "b"]; levels = ["a", "b", "z"])
        @test validate_mixture_against_data(mix("z" => 1.0), unused, :g) === true

        # Bool: false and true, as Bool or as strings, whether or not both occur
        tr = [true, false, true, false]
        @test validate_mixture_against_data(mix(true => 0.6, false => 0.4), tr, :treated) === true
        @test validate_mixture_against_data(mix("true" => 0.3, "false" => 0.7), tr, :treated) === true
        @test validate_mixture_against_data(mix(true => 0.5, false => 0.5), [true, true], :treated) === true
        @test_throws ArgumentError validate_mixture_against_data(mix("yes" => 0.5, "no" => 0.5), tr, :treated)

        # other columns: the values that occur
        region = ["north", "south", "east", "west"]
        @test validate_mixture_against_data(mix("north" => 0.25, "south" => 0.75), region, :region) === true
        @test_throws ArgumentError validate_mixture_against_data(mix("north" => 0.5, "central" => 0.5), region, :region)
    end

    @testset "mixture_to_scenario_value" begin
        # CategoricalArray: share-weighted level index, levels sorted (college=1, graduate=2, high_school=3)
        edu = categorical(["high_school", "college", "graduate"])
        @test mixture_to_scenario_value(mix("high_school" => 0.5, "college" => 0.3, "graduate" => 0.2), edu) ==
              0.5 * 3 + 0.3 * 1 + 0.2 * 2
        @test mixture_to_scenario_value(mix("college" => 1.0), edu) == 1.0
        # the column's level order, not alphabetical order
        rev = categorical(["a", "b", "c"]; levels = ["c", "b", "a"])
        @test mixture_to_scenario_value(mix("a" => 0.75, "c" => 0.25), rev) == 0.75 * 3 + 0.25 * 1

        # Bool: the weight of true, whether the levels are Bool or strings
        tr = [true, false, true]
        @test mixture_to_scenario_value(mix(true => 0.3, false => 0.7), tr) == 0.3
        @test mixture_to_scenario_value(mix("true" => 0.8, "false" => 0.2), tr) == 0.8
        @test mixture_to_scenario_value(mix(true => 1.0, false => 0.0), tr) == 1.0
        @test mixture_to_scenario_value(mix(true => 0.0, false => 1.0), tr) == 0.0
        @test mixture_to_scenario_value(mix(false => 1.0), tr) == 0.0
        @test_throws ArgumentError mixture_to_scenario_value(mix("A" => 0.5, "B" => 0.5), tr)

        # other columns: share-weighted index among the sorted unique values (A=1, B=2, C=3)
        @test mixture_to_scenario_value(mix("A" => 0.2, "B" => 0.3, "C" => 0.5), ["B", "C", "A", "A", "B"]) ==
              0.2 * 1 + 0.3 * 2 + 0.5 * 3

        # MixtureWithLevels delegates to its mixture (it used to call itself until the stack overflowed)
        mwl = MixtureWithLevels(mix("a" => 0.75, "c" => 0.25), ["c", "b", "a"])
        @test mixture_to_scenario_value(mwl, rev) == 0.75 * 3 + 0.25 * 1
    end

    @testset "CategoricalArray columns without CategoricalArrays loaded in Main" begin
        # The CategoricalArray branch used to require `isdefined(Main, :CategoricalArrays)`;
        # without it a column fell back to its sorted unique values instead of its levels
        code = """
            using FormulaCompiler
            col = FormulaCompiler.CategoricalArrays.categorical(["c", "a", "b"]; levels = ["c", "a", "b"])
            print(mixture_to_scenario_value(mix("a" => 1.0), col), " ",
                  isdefined(Main, :CategoricalArrays))
            """
        out = read(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`, String)
        @test out == "2.0 false"
    end
end
