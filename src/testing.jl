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

struct LMTest
    name::String
    formula::FormulaTerm
end

linear_formulas = [
    LMTest("Intercept only", @formula(continuous_response ~ 1)),
    LMTest("No intercept", @formula(continuous_response ~ 0 + x)),
    LMTest("Simple continuous", @formula(continuous_response ~ x)),
    LMTest("Simple categorical", @formula(continuous_response ~ group3)),
    LMTest("Multiple continuous", @formula(continuous_response ~ x + y)),
    LMTest("Multiple categorical", @formula(continuous_response ~ group3 + group4)),
    LMTest("Mixed", @formula(continuous_response ~ x + group3)),
    LMTest("Simple interaction", @formula(continuous_response ~ x * group3)),
    LMTest("Interaction w/o main", @formula(continuous_response ~ x & group3)),
    LMTest("Function", @formula(continuous_response ~ log(z))),
    LMTest("Three-way interaction", @formula(continuous_response ~ x * y * group3)),
    LMTest("Four-way interaction", @formula(continuous_response ~ x * y * group3 * group4)),
    LMTest("Four-way w/ function", @formula(continuous_response ~ exp(x) * y * group3 * group4)),
    LMTest("Complex interaction", @formula(continuous_response ~ x * y * group3 + log(z) * group4)),
]

struct GLMTest
    name::String
    formula::FormulaTerm
    distribution
    link
end

glm_tests = [
    GLMTest("Logistic: simple", @formula(logistic_response ~ x), Binomial(), LogitLink()),
    GLMTest("Logistic: mixed", @formula(logistic_response ~ x + group3), Binomial(), LogitLink()),
    GLMTest("Logistic: interaction", @formula(logistic_response ~ x * group3), Binomial(), LogitLink()),
    GLMTest("Logistic: function", @formula(logistic_response ~ log(abs(z)) + group3), Binomial(), LogitLink()),
    GLMTest("Logistic: complex", @formula(logistic_response ~ x * y * group3 + log(abs(z)) + group4), Binomial(), LogitLink()),
    GLMTest("Poisson: simple", @formula(count_response ~ x), Poisson(), LogLink()),
    GLMTest("Poisson: mixed", @formula(count_response ~ x + group3), Poisson(), LogLink()),
    GLMTest("Poisson: interaction", @formula(count_response ~ x * group3), Poisson(), LogLink()),
    GLMTest("Gamma: mixed", @formula(z ~ x + group3), Gamma(), LogLink()),
    GLMTest("Gaussian: mixed", @formula(z ~ x + group3), Normal(), LogLink()),
]

struct LMMTest
    name::String
    formula::FormulaTerm
end

lmm_formulas = [
    LMMTest("Random intercept", @formula(continuous_response ~ x + (1|subject))),
    LMMTest("Mixed + categorical", @formula(continuous_response ~ x + group3 + (1|subject))),
    LMMTest("Random slope", @formula(continuous_response ~ x + (x|subject))),
    LMMTest("Random slope + cat", @formula(continuous_response ~ x + group3 + (x|subject))),
    LMMTest("Multiple random", @formula(continuous_response ~ x + (1|subject) + (1|cluster))),
    LMMTest("Interaction + random", @formula(continuous_response ~ x * group3 + (1|subject))),
]

struct GLMMTest
    name::String
    formula::FormulaTerm
    distribution
    link
end

glmm_tests = [
    GLMMTest("Logistic: 1", @formula(logistic_response ~ x + (1|subject)), Binomial(), LogitLink()),
    GLMMTest("Logistic: 2", @formula(logistic_response ~ x + group3 + (1|subject)), Binomial(), LogitLink()),
    GLMMTest("Poisson: 1", @formula(count_response ~ x + (1|subject)), Poisson(), LogLink()),
    GLMMTest("Poisson: 2", @formula(count_response ~ x + group3 + (1|cluster)), Poisson(), LogLink()),
]

test_formulas = (
    lm = linear_formulas,
    glm = glm_tests,
    lmm = lmm_formulas,
    glmm = glmm_tests
)
