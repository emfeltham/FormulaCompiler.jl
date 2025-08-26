# Debug four-way interaction with full test data
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays
using FormulaCompiler: make_test_data, test_formulas
using Random

Random.seed!(08540)

# Use the same test data as test_models.jl
n = 500
df = make_test_data(; n)
data = Tables.columntable(df)

println("="^70)
println("TESTING FOUR-WAY INTERACTIONS WITH FULL DATA")
println("="^70)

# Find the four-way interaction formulas
fourway_formulas = filter(fx -> fx.name == "Four-way interaction", test_formulas.lm)
fourway_function_formulas = filter(fx -> fx.name == "Four-way w/ function", test_formulas.lm)

if !isempty(fourway_formulas)
    println("\nTesting: Four-way interaction")
    fx = fourway_formulas[1]
    model = lm(fx.formula, df)
    compiled = compile_formula(model, data)
    expected_matrix = modelmatrix(model)
    
    println("Formula: ", fx.formula)
    println("Model matrix size: ", size(expected_matrix))
    println("Compiled output size: ", length(compiled))
    
    # Test specific rows
    output = Vector{Float64}(undef, length(compiled))
    test_rows = [1, 10, 50, 100, 250, 500]
    
    failures = Int[]
    for test_row in test_rows
        compiled(output, data, test_row)
        expected = expected_matrix[test_row, :]
        
        if !isapprox(output, expected, rtol=1e-10)
            push!(failures, test_row)
            println("\n❌ Row $test_row failed")
            
            # Find mismatched positions
            diff = abs.(output .- expected)
            problem_positions = findall(diff .> 1e-10)
            
            if length(problem_positions) <= 10
                for pos in problem_positions
                    col_name = coefnames(model)[pos]
                    println("  Position $pos ($col_name):")
                    println("    Expected: $(expected[pos])")
                    println("    Got: $(output[pos])")
                    println("    Diff: $(diff[pos])")
                end
            else
                println("  Too many mismatches ($(length(problem_positions)))")
                # Show first few
                for pos in problem_positions[1:min(5, length(problem_positions))]
                    col_name = coefnames(model)[pos]
                    println("  Position $pos ($col_name): expected=$(expected[pos]), got=$(output[pos])")
                end
            end
        else
            println("✅ Row $test_row passed")
        end
    end
    
    if !isempty(failures)
        println("\n$(length(failures)) rows failed: ", failures)
        
        # Let's analyze the pattern
        println("\nAnalyzing failure pattern...")
        first_fail = failures[1]
        println("First failure at row $first_fail:")
        println("  x = $(df.x[first_fail])")
        println("  y = $(df.y[first_fail])")
        println("  group3 = $(df.group3[first_fail])")
        println("  group4 = $(df.group4[first_fail])")
    end
end

if !isempty(fourway_function_formulas)
    println("\n" * "="^70)
    println("Testing: Four-way w/ function")
    fx = fourway_function_formulas[1]
    model = lm(fx.formula, df)
    compiled = compile_formula(model, data)
    expected_matrix = modelmatrix(model)
    
    println("Formula: ", fx.formula)
    println("Model matrix size: ", size(expected_matrix))
    println("Compiled output size: ", length(compiled))
    
    # Test specific rows
    output = Vector{Float64}(undef, length(compiled))
    test_rows = [1, 10, 50, 100, 250, 500]
    
    failures = Int[]
    for test_row in test_rows
        compiled(output, data, test_row)
        expected = expected_matrix[test_row, :]
        
        if !isapprox(output, expected, rtol=1e-10)
            push!(failures, test_row)
        else
            println("✅ Row $test_row passed")
        end
    end
    
    if !isempty(failures)
        println("\n❌ $(length(failures)) rows failed: ", failures)
    end
end