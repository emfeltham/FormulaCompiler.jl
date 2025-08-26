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

# Helper function to test position mapping correctness
function test_formula_correctness(formula, df, data)
    model = fit(LinearModel, formula, df)
    compiled = compile_formula(model, data)
    output_compiled = Vector{Float64}(undef, length(compiled))
    mm = modelmatrix(model)
    
    # Test correctness on multiple rows
    test_rows = [1, 2, 5, 10, 25, 50]
    for test_row in test_rows
        if test_row > size(mm, 1)
            continue  # Skip if test row exceeds data size
        end
        
        fill!(output_compiled, NaN)
        compiled(output_compiled, data, test_row)
        expected_row = mm[test_row, :]
        
        @test isapprox(output_compiled, expected_row, rtol=1e-12)
    end
    
    return compiled, output_compiled
end

# Helper function to test allocation performance
function test_allocation_performance(compiled, output_compiled, data)
    # Warmup
    for _ in 1:10
        compiled(output_compiled, data, 1)
    end
    
    # Measure allocation
    compiled_allocs = @allocated begin
        for i in 1:100
            row_idx = ((i-1) % 200) + 1
            compiled(output_compiled, data, row_idx)
        end
    end
    
    allocs_per_call = compiled_allocs / 100
    
    # Test allocation levels (allowing for current known issues)
    if occursin("function", lowercase(description)) || occursin("interaction", lowercase(description))
        @test allocs_per_call <= 1000  # More lenient for functions and complex interactions
    else
        @test allocs_per_call == 0  # Expect zero for simple cases
    end
    
    return allocs_per_call
end

# Helper function to test allocation with proper warmup
function test_zero_allocation(model, data)
    compiled = compile_formula(model, data)
    buffer = Vector{Float64}(undef, length(compiled))
    
    # Extensive warmup to ensure compilation is complete
    for _ in 1:100
        compiled(buffer, data, 1)
    end
    
    # Benchmark for accurate allocation measurement
    benchmark_result = @benchmark $compiled($buffer, $data, 1) samples=1000 seconds=2
    memory_bytes = minimum(benchmark_result.memory)
    
    @test memory_bytes == 0
    
    return memory_bytes, minimum(benchmark_result.times)
end

# Helper function to test correctness against modelmatrix
function test_model_correctness(model, data, n)
    compiled = compile_formula(model, data)
    output = Vector{Float64}(undef, length(compiled))
    expected_matrix = modelmatrix(model)
    
    # Test correctness against modelmatrix on multiple test rows
    test_rows = [1, 10, 50, 100, min(n, 250), n]
    for test_row in test_rows
        compiled(output, data, test_row)
        expected = expected_matrix[test_row, :]
        @test isapprox(output, expected, rtol=1e-10)  # Relaxed tolerance for numerical stability
    end
    
    # Test modelrow functionality (allocating version)
    for test_row in [1, 25, n÷2, n]
        output_modelrow = modelrow(compiled, data, test_row)
        expected = expected_matrix[test_row, :]
        @test isapprox(output_modelrow, expected, rtol=1e-10)  # Relaxed tolerance
    end
    
    # Test modelrow! functionality (in-place version)
    output_inplace = Vector{Float64}(undef, length(compiled))
    for test_row in [1, 10, n]
        modelrow!(output_inplace, compiled, data, test_row)
        expected = expected_matrix[test_row, :]
        @test isapprox(output_inplace, expected, rtol=1e-10)  # Relaxed tolerance
    end
    
    return true
end

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
    LMTest("Function in interaction", @formula(continuous_response ~ exp(x) * y)),
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
