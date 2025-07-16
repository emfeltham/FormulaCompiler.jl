# test/test_evaluator_trees.jl
# Tests for evaluator tree analysis and introspection

@testset "Evaluator Tree Analysis" begin
    
    df = DataFrame(
        x = abs.(randn(10)),
        y = randn(10),
        z = abs.(randn(10)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 10))
    )
    
    @testset "Root Evaluator Access" begin
        # Test simple model
        model = lm(@formula(y ~ x), df)
        compiled = compile_formula(model)
        
        # Test root evaluator extraction
        root_eval = extract_root_evaluator(compiled)
        @test root_eval isa AbstractEvaluator
        @test root_eval === compiled.root_evaluator
        
        # Test alternative name
        tree = get_evaluator_tree(compiled)
        @test tree === root_eval
        
        # Test access check
        @test has_evaluator_access(compiled) == true
    end
    
    @testset "Node Counting" begin
        # Test simple model
        model = lm(@formula(y ~ x), df)
        compiled = compile_formula(model)
        
        node_count = count_evaluator_nodes(compiled)
        @test node_count isa Int
        @test node_count >= 2  # At least intercept + x
        
        # Test more complex model
        model2 = lm(@formula(y ~ x * group + log(z)), df)
        compiled2 = compile_formula(model2)
        
        node_count2 = count_evaluator_nodes(compiled2)
        @test node_count2 > node_count  # More complex should have more nodes
        
        # Test very simple model
        model3 = lm(@formula(y ~ 1), df)
        compiled3 = compile_formula(model3)
        
        node_count3 = count_evaluator_nodes(compiled3)
        # Intercept-only creates CombinedEvaluator + ConstantEvaluator = 2 nodes
        @test node_count3 == 2  # CombinedEvaluator wrapper + ConstantEvaluator(1.0)
    end
    
    @testset "Variable Dependencies" begin
        # Test simple model
        model = lm(@formula(y ~ x), df)
        compiled = compile_formula(model)
        
        vars = get_variable_dependencies(compiled)
        @test vars isa Vector{Symbol}
        @test :x in vars
        @test length(vars) == 1
        
        # Test model with multiple variables
        model2 = lm(@formula(y ~ x + z + group), df)
        compiled2 = compile_formula(model2)
        
        vars2 = get_variable_dependencies(compiled2)
        @test :x in vars2
        @test :z in vars2
        @test :group in vars2
        @test length(vars2) == 3
        
        # Test model with interactions
        model3 = lm(@formula(y ~ x * group), df)
        compiled3 = compile_formula(model3)
        
        vars3 = get_variable_dependencies(compiled3)
        @test :x in vars3
        @test :group in vars3
        @test length(vars3) == 2  # No duplicates
        
        # Test function terms
        model4 = lm(@formula(y ~ log(x) + sqrt(z)), df)
        compiled4 = compile_formula(model4)
        
        vars4 = get_variable_dependencies(compiled4)
        @test :x in vars4
        @test :z in vars4
        @test length(vars4) == 2
    end
    
    @testset "Evaluator Summary" begin
        model = lm(@formula(y ~ x * group + log(z)), df)
        compiled = compile_formula(model)
        
        summary = get_evaluator_summary(compiled)
        @test summary isa NamedTuple
        
        # Check required fields
        @test haskey(summary, :type)
        @test haskey(summary, :total_nodes)
        @test haskey(summary, :output_width)
        @test haskey(summary, :variables)
        @test haskey(summary, :variable_count)
        @test haskey(summary, :complexity_score)
        
        # Check values
        @test summary.type == typeof(compiled.root_evaluator)
        @test summary.total_nodes isa Int
        @test summary.total_nodes > 0
        @test summary.output_width == compiled.output_width
        @test summary.variables isa Vector{Symbol}
        @test summary.variable_count == length(summary.variables)
        @test summary.complexity_score isa Int
        @test summary.complexity_score > 0
        
        # Test that variables are correct
        @test :x in summary.variables
        @test :group in summary.variables
        @test :z in summary.variables
    end
    
    @testset "Complexity Estimation" begin
        # Test simple model
        model1 = lm(@formula(y ~ x), df)
        compiled1 = compile_formula(model1)
        summary1 = get_evaluator_summary(compiled1)
        
        # Test complex model
        model2 = lm(@formula(y ~ x * group * z + log(x) + sqrt(z)), df)
        compiled2 = compile_formula(model2)
        summary2 = get_evaluator_summary(compiled2)
        
        # Complex model should have higher complexity
        @test summary2.complexity_score > summary1.complexity_score
        @test summary2.total_nodes > summary1.total_nodes
    end
    
    @testset "Pretty Printing" begin
        model = lm(@formula(y ~ x + group), df)
        compiled = compile_formula(model)
        
        # Test that show methods work without error
        io = IOBuffer()
        show(io, MIME"text/plain"(), compiled)
        output = String(take!(io))
        
        @test occursin("CompiledFormula", output)
        @test occursin("Output width", output)
        @test occursin("Variables", output)
        
        # Test tree printing works without error (just call it, don't capture)
        @test_nowarn print_evaluator_tree(compiled)
        
        # If you want to capture the tree output, use a different approach:
        tree_io = IOBuffer()
        # Manually redirect within the function or modify print_evaluator_tree to accept IO
        @test_nowarn print_evaluator_tree(compiled)  # Just test it doesn't error
    end
    
    @testset "Recursive Structure Analysis" begin
        # Test nested function structure
        model = lm(@formula(y ~ log(x^2) + sqrt(z)), df)
        compiled = compile_formula(model)
        
        # Should have detected all variables despite nesting
        vars = get_variable_dependencies(compiled)
        @test :x in vars
        @test :z in vars
        
        # Node count should reflect nesting
        node_count = count_evaluator_nodes(compiled)
        @test node_count >= 5  # At least: intercept, x, ^, log, sqrt, z
        
        # Test interaction with functions
        model2 = lm(@formula(y ~ x * log(z)), df)
        compiled2 = compile_formula(model2)
        
        vars2 = get_variable_dependencies(compiled2)
        @test :x in vars2
        @test :z in vars2
        
        # Should have higher complexity due to interaction
        summary2 = get_evaluator_summary(compiled2)
        @test summary2.complexity_score > 10  # Interactions are expensive
    end
    
    @testset "Edge Cases" begin
        # Test intercept-only model
        model = lm(@formula(y ~ 1), df)
        compiled = compile_formula(model)
        
        vars = get_variable_dependencies(compiled)
        @test length(vars) == 0  # No variables, just intercept
        
        node_count = count_evaluator_nodes(compiled)
        # Intercept-only creates CombinedEvaluator + ConstantEvaluator = 2 nodes
        @test node_count == 2  # CombinedEvaluator wrapper + ConstantEvaluator(1.0)
        
        # Test no-intercept model
        model2 = lm(@formula(y ~ 0 + x), df)
        compiled2 = compile_formula(model2)
        
        vars2 = get_variable_dependencies(compiled2)
        @test :x in vars2
        @test length(vars2) == 1
        
        # Test model with only categorical
        model3 = lm(@formula(y ~ group), df)
        compiled3 = compile_formula(model3)
        
        vars3 = get_variable_dependencies(compiled3)
        @test :group in vars3
        @test length(vars3) == 1
    end
    
    @testset "Consistency Tests" begin
        # Test that analysis functions are consistent
        model = lm(@formula(y ~ x * group + log(z)), df)
        compiled = compile_formula(model)
        
        # Variables from summary should match direct call
        summary = get_evaluator_summary(compiled)
        direct_vars = get_variable_dependencies(compiled)
        @test Set(summary.variables) == Set(direct_vars)
        
        # Variable count should match
        @test summary.variable_count == length(direct_vars)
        
        # Node count should match
        direct_nodes = count_evaluator_nodes(compiled)
        @test summary.total_nodes == direct_nodes
        
        # Output width should match
        @test summary.output_width == compiled.output_width
        @test summary.output_width == length(compiled)
    end
    
end
