# testing.jl

function make_test_data(; n = 500)
    df = DataFrame(
        # Continuous variables
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,  # Positive for log
        w = randn(n),
        t = randn(n),
        
        # Categorical variables with different levels
        group3 = categorical(rand(["A", "B", "C"], n)),           # 3 levels
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      # 4 levels
        binary = categorical(rand(["Yes", "No"], n)),             # 2 levels
        group5 = categorical(rand(["P", "Q", "R", "S", "T"], n)), # 5 levels
        
        # Random effects grouping variables
        subject = categorical(rand(1:20, n)),     # 20 subjects
        cluster = categorical(rand(1:10, n)),     # 10 clusters
        
        # Boolean/logical
        flag = rand([true, false], n),
        
        # Response variables for different model types
        continuous_response = randn(n),
        binary_response = rand([0, 1], n),
        count_response = rand(0:10, n),
        
    )
    # Create correlated response for logistic
    df.linear_predictor = 0.5 .+ 0.3 .* randn(n) .+ 0.2 .* (df.group3 .== "A")
    
    # Create logistic response from linear predictor
    probabilities = 1 ./ (1 .+ exp.(-df.linear_predictor))
    df.logistic_response = [rand() < p ? 1 : 0 for p in probabilities]
    return df
end

###############################################################################
# TESTING
###############################################################################

function test_cases(cases, df, data)
    if typeof(cases) <: Tuple
        cases = [cases]
    end
    println("Model scenario testing")
    for (f, nm) in cases
        # fit
        model = fit(LinearModel, f, df);
        # allocate output
        output_after = Vector{Float64}(undef, size(modelmatrix(model), 2));
        # compile
        compiled_after = compile_formula(model, data);
        compiled_after(output_after, data, 1);

        println("Case: " * nm)
        @btime $compiled_after($output_after, $data, $1);
    end
end

function test_correctness(cases, df, data; i = 1)
    # normalize to a Vector
    cases = isa(cases, Tuple) ? [cases] : collect(cases)

    @testset "Model-scenario testing" begin
        for (f, nm) in cases
            @testset "$nm" begin
                # fit the model
                model = fit(LinearModel, f, df)
                mm = modelmatrix(model);
                mr = mm[i, :]
                # prepare your “after” vector
                output_after = Vector{Float64}(undef, size(mm, 2))
                # compile
                compiled_after = compile_formula(model, data)
                
                # run it
                compiled_after(output_after, data, i)
                # now the actual test
                @test isapprox(mr, output_after; atol = 1e-5)
            end
        end
    end
end

function test_data(; n = 200)
    # Create test data with all variable types
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,  # Positive for log
        w = randn(n),
        t = randn(n),
        group2 = categorical(rand(["Z", "M", "L"], n)),
        group3 = categorical(rand(["A", "B", "C"], n)),
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),
        binary = categorical(rand(["Yes", "No"], n)),
        group5 = categorical(rand(["P", "Q", "R", "S", "T"], n)),
        response = randn(n)
    )
    data = Tables.columntable(df)
    return df, data
end

export test_cases, test_correctness, test_data

###############################################################################
# MODEL SCENARIOS
###############################################################################

test_basic = [
    (@formula(response ~ 1), "Baseline (no interactions)"),
    (@formula(response ~ x), "Baseline (no interactions)"),
    (@formula(response ~ x + y), "Multiple continuous"),
    (@formula(response ~ x + y + z + w + t),  "Many continuous variables"),
];

test_categoricals = [
    (@formula(response ~ group3), "Single categorical"),
    (@formula(response ~ group3 + group4), "Two categoricals"),
    (@formula(response ~ group3 + group4 + binary), "Three categoricals"),
    (@formula(response ~ x + group3), "Mixed continuous + categorical"),
    (@formula(response ~ x + y + group3 + group4), "Multiple mixed"),
];

test_functions = [
    (@formula(response ~ log(z)), "Function 1"),
    (@formula(response ~ z > 0), "Comparison 1"),
    (@formula(response ~ x^2), "Function 2"),
    (@formula(response ~ log(abs(z))), "2-Nested Function 1"),
    (@formula(response ~ abs(z)^2), "2-Nested Function 2"),
    (@formula(response ~ abs(log(abs(z)))), "3-Nested Function"),
    (@formula(response ~ log(abs(z))^2), "3-Nested Function 2"),
    (@formula(response ~ (z + y)^2), "2-Function"),
    (@formula(response ~ abs(z + x)), "2-Function 1"),
    (@formula(response ~ abs(z + y + x)), "3-Function 1"),
    (@formula(response ~ (z + y + x)^2), "3-Function 2"),
    (@formula(response ~ abs(z + y + x + w)), "3-Function 1")
];

test_interactions = [
    (@formula(response ~ x * y), "Simple 2-way interaction 1"),
    (@formula(response ~ x + x & y), "Simple 2-way interaction 2"),
    (@formula(response ~ x * group3), "Continuous × Categorical 1"),
    (@formula(response ~ x & group3 + x), "Continuous × Categorical 2"),
    (@formula(response ~ group3 * binary), "Categorical × Categorical"),
    (@formula(response ~ log(z) * group4), "Function × Categorical"),
    (@formula(response ~ x * log(z)), "Continuous × Function"),
    (@formula(response ~ log(abs(x) + 2) * (y - 3.5)), "2-way Complex Function 1"),

    (@formula(response ~ x * y * z), "3-way continuous"),
    (@formula(response ~ x * y * group3), "3-way interaction"),
    (@formula(response ~ group3 * group4 * binary), "3-way categorical, binary"),
    (@formula(response ~ group3 * group4 * group5), "3-way categorical"),
    
    (@formula(response ~ x * y * z * w), "4-way interaction"),
    (@formula(response ~ group2 * group3 * group4 * group5), "4-way categorical"),
    (@formula(response ~ log(z) * exp(w) * group3), "Multiple functions × categorical"),
    (@formula(response ~ x * y * group3 + log(z) * group4), "Your original formula!"),
];

test_scenarios = (
    basic = test_basic, categoricals = test_categoricals, functions = test_functions, interactions = test_interactions, 
)

export test_scenarios
