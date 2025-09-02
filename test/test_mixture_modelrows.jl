# Test correctness of modelrow functions with categorical mixtures
using Test
using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays
using BenchmarkTools

# Test mixture object (consistent with other test files)
struct TestMixture
    levels::Vector{String}
    weights::Vector{Float64}
    
    function TestMixture(levels::Vector{String}, weights::Vector{Float64})
        @assert length(levels) == length(weights)
        @assert isapprox(sum(weights), 1.0, atol=1e-10)
        @assert all(w >= 0 for w in weights)
        new(levels, weights)
    end
end

@testset "Modelrow Functions with Categorical Mixtures" begin
    
    # Create training data with normal categorical (for GLM fitting)
    train_df = DataFrame(
        y = randn(50),
        x = randn(50),
        z = randn(50),
        group = categorical(rand(["A", "B", "C"], 50))
    )
    
    @testset "Basic Modelrow Functions" begin
        # Fit simple interaction model
        model = lm(@formula(y ~ x * group), train_df)
        
        # Create test data with mixtures
        test_df = DataFrame(
            x = [1.0, 2.0, 3.0],
            group = [TestMixture(["A", "B"], [0.3, 0.7]),
                     TestMixture(["A", "B"], [0.3, 0.7]),
                     TestMixture(["A", "B"], [0.3, 0.7])]
        )
        
        compiled = compile_formula(model, Tables.columntable(test_df))
        
        @testset "All Modelrow Interfaces Equivalent" begin
            for row_idx in 1:3
                # Test all three modelrow interfaces
                output1 = Vector{Float64}(undef, length(compiled))
                output2 = Vector{Float64}(undef, length(compiled))
                
                # Direct compiled call
                compiled(output1, Tables.columntable(test_df), row_idx)
                
                # modelrow! function
                modelrow!(output2, compiled, Tables.columntable(test_df), row_idx)
                
                # modelrow convenience function
                output3 = modelrow(compiled, Tables.columntable(test_df), row_idx)
                
                # All should produce identical results
                @test output1 ≈ output2 rtol=1e-12
                @test output2 ≈ output3 rtol=1e-12
                @test length(output3) == length(compiled)
            end
        end
        
        @testset "Zero Allocation Performance" begin
            output = Vector{Float64}(undef, length(compiled))
            data_ct = Tables.columntable(test_df)
            
            # Benchmark direct compiled call
            bench1 = @benchmark $compiled($output, $data_ct, 1) samples=1000 evals=1
            
            # Benchmark modelrow! call  
            bench2 = @benchmark modelrow!($output, $compiled, $data_ct, 1) samples=1000 evals=1
            
            # Test allocation (should be zero after warmup)
            @test bench1.allocs == 0
            @test bench2.allocs == 0
        end
    end
    
    @testset "Mathematical Correctness" begin
        # Create deterministic model for predictable results
        simple_train = DataFrame(
            y = [1.0, 2.0, 1.0, 2.0],
            x = [1.0, 1.0, 1.0, 1.0], 
            group = categorical(["A", "B", "A", "B"])
        )
        
        model = lm(@formula(y ~ x + group), simple_train)
        
        @testset "Binary Mixture vs Pure Categories" begin
            # Test with 50-50 mixture
            mixture_df = DataFrame(
                x = [1.0],
                group = [TestMixture(["A", "B"], [0.5, 0.5])]
            )
            
            # Create data with full categorical structure (both levels present)
            full_structure_df = DataFrame(
                x = [1.0, 1.0],
                group = categorical(["A", "B"])  # Full structure needed for correct contrasts
            )
            
            # Compile both
            mixture_compiled = compile_formula(model, Tables.columntable(mixture_df))
            full_compiled = compile_formula(model, Tables.columntable(full_structure_df))
            
            # Evaluate
            mixture_result = modelrow(mixture_compiled, Tables.columntable(mixture_df), 1)
            pure_a_result = modelrow(full_compiled, Tables.columntable(full_structure_df), 1)  # Row 1 = A
            pure_b_result = modelrow(full_compiled, Tables.columntable(full_structure_df), 2)  # Row 2 = B
            
            # Mixture should equal weighted average of pure results
            expected_result = 0.5 .* pure_a_result .+ 0.5 .* pure_b_result
            @test mixture_result ≈ expected_result rtol=1e-12
        end
        
        @testset "Asymmetric Mixture Weights" begin
            # Test 80-20 mixture
            weights = [0.8, 0.2]
            mixture_df = DataFrame(
                x = [2.0],
                group = [TestMixture(["A", "B"], weights)]
            )
            
            # Full structure for correct contrast evaluation
            full_structure_df = DataFrame(
                x = [2.0, 2.0],
                group = categorical(["A", "B"])
            )
            
            mixture_compiled = compile_formula(model, Tables.columntable(mixture_df))
            full_compiled = compile_formula(model, Tables.columntable(full_structure_df))
            
            mixture_result = modelrow(mixture_compiled, Tables.columntable(mixture_df), 1)
            pure_a_result = modelrow(full_compiled, Tables.columntable(full_structure_df), 1)  # Row 1 = A
            pure_b_result = modelrow(full_compiled, Tables.columntable(full_structure_df), 2)  # Row 2 = B
            
            expected_result = weights[1] .* pure_a_result .+ weights[2] .* pure_b_result
            @test mixture_result ≈ expected_result rtol=1e-12
        end
    end
    
    @testset "Derivative Modelrow Functions" begin
        # Complex interaction model for derivatives
        model = lm(@formula(y ~ x * z * group), train_df)
        
        test_df = DataFrame(
            x = [1.5, 2.5],
            z = [0.5, 1.5],
            group = [TestMixture(["A", "B"], [0.4, 0.6]),
                     TestMixture(["A", "B"], [0.4, 0.6])]
        )
        
        compiled = compile_formula(model, Tables.columntable(test_df))
        continuous_vars = continuous_variables(compiled, Tables.columntable(test_df))
        
        @test !isempty(continuous_vars)  # Should have x and z
        
        de = build_derivative_evaluator(compiled, Tables.columntable(test_df); vars=continuous_vars)
        
        @testset "Marginal Effects with Mixtures" begin
            gradient_fd = Vector{Float64}(undef, length(continuous_vars))
            gradient_ad = Vector{Float64}(undef, length(continuous_vars))
            
            for row_idx in 1:2
                # Test both FD and AD backends
                marginal_effects_eta!(gradient_fd, de, coef(model), row_idx; backend=:fd)
                marginal_effects_eta!(gradient_ad, de, coef(model), row_idx; backend=:ad)
                
                # Should have derivatives for both x and z
                @test length(gradient_fd) == length(continuous_vars)
                @test all(isfinite, gradient_fd)
                @test length(gradient_ad) == length(continuous_vars)
                @test all(isfinite, gradient_ad)
                
                # Both backends should produce non-zero results for this interaction model
                @test !all(iszero, gradient_fd)
                @test !all(iszero, gradient_ad)
            end
            
            # Test that backends agree on simpler cases by testing with a simpler model
            simple_model = lm(@formula(y ~ x + z), train_df[1:20, :])  # Simpler model
            simple_test_df = DataFrame(x = [1.0], z = [1.0], 
                                     group = [TestMixture(["A", "B"], [0.5, 0.5])])
            simple_compiled = compile_formula(simple_model, Tables.columntable(simple_test_df))
            simple_vars = continuous_variables(simple_compiled, Tables.columntable(simple_test_df))
            
            if !isempty(simple_vars)
                simple_de = build_derivative_evaluator(simple_compiled, Tables.columntable(simple_test_df); vars=simple_vars)
                simple_grad_fd = Vector{Float64}(undef, length(simple_vars))
                simple_grad_ad = Vector{Float64}(undef, length(simple_vars))
                
                marginal_effects_eta!(simple_grad_fd, simple_de, coef(simple_model), 1; backend=:fd)
                marginal_effects_eta!(simple_grad_ad, simple_de, coef(simple_model), 1; backend=:ad)
                
                # Simple models should have good FD vs AD agreement
                @test simple_grad_fd ≈ simple_grad_ad rtol=1e-6
            end
        end
        
        @testset "Zero Allocation Derivatives (FD)" begin
            gradient = Vector{Float64}(undef, length(continuous_vars))
            coeffs = coef(model)
            
            # Benchmark FD derivatives
            bench_fd = @benchmark marginal_effects_eta!($gradient, $de, $coeffs, 1; backend=:fd) samples=1000 evals=1
            
            # Test zero allocation for FD backend
            @test bench_fd.allocs == 0
        end
    end
    
    @testset "Multi-level Mixtures" begin
        # Test with 3-level mixture
        model = lm(@formula(y ~ x * group), train_df)
        
        test_df = DataFrame(
            x = [1.0],
            group = [TestMixture(["A", "B", "C"], [0.2, 0.3, 0.5])]
        )
        
        compiled = compile_formula(model, Tables.columntable(test_df))
        result = modelrow(compiled, Tables.columntable(test_df), 1)
        
        @test length(result) == length(compiled)
        @test all(isfinite, result)
        
        # Verify against manual weighted combination using full structure
        full_3level_df = DataFrame(
            x = [1.0, 1.0, 1.0],
            group = categorical(["A", "B", "C"])  # All three levels present
        )
        full_3level_compiled = compile_formula(model, Tables.columntable(full_3level_df))
        
        pure_a_result = modelrow(full_3level_compiled, Tables.columntable(full_3level_df), 1)  # A
        pure_b_result = modelrow(full_3level_compiled, Tables.columntable(full_3level_df), 2)  # B  
        pure_c_result = modelrow(full_3level_compiled, Tables.columntable(full_3level_df), 3)  # C
        
        weights = [0.2, 0.3, 0.5]
        expected = weights[1] .* pure_a_result .+ weights[2] .* pure_b_result .+ weights[3] .* pure_c_result
        @test result ≈ expected rtol=1e-12
    end
    
end