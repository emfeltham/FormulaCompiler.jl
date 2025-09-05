# Basic tests for categorical mixture detection (Phase 1)
# Tests the core mixture detection utilities added in Phase 1 implementation

using Test
using FormulaCompiler
using DataFrames, Tables
using StatsModels, GLM

# Mock mixture object for testing
# This simulates the structure expected from Margins.jl's mix() function
struct MockMixture
    levels::Vector{String}
    weights::Vector{Float64}
    
    function MockMixture(levels, weights; validate=true)
        if length(levels) != length(weights)
            error("Levels and weights must have same length")
        end
        if validate && !isapprox(sum(weights), 1.0, atol=1e-10)
            error("Weights must sum to 1.0")
        end
        if validate && any(w < 0 for w in weights)
            error("Weights must be non-negative")
        end
        if validate && length(unique(levels)) != length(levels)
            error("Duplicate levels not allowed")
        end
        new(levels, weights)
    end
end

# Helper to create invalid mixtures for testing
function MockMixture(levels, weights)
    MockMixture(levels, weights; validate=true)
end

function unsafe_mock_mixture(levels, weights)
    MockMixture(levels, weights; validate=false)
end

# Helper function to create mock mixture (simulates Margins.jl's mix function)
mock_mix(pairs...) = MockMixture([string(p.first) for p in pairs], [p.second for p in pairs])

@testset "Mixture Detection Utilities" begin
    @testset "is_mixture_column" begin
        # Test with mixture column
        mixture_obj = mock_mix("A" => 0.3, "B" => 0.7)
        mixture_col = [mixture_obj, mixture_obj, mixture_obj]
        @test FormulaCompiler.is_mixture_column(mixture_col) == true
        
        # Test with regular categorical data
        regular_col = ["A", "B", "A", "B"]
        @test FormulaCompiler.is_mixture_column(regular_col) == false
        
        # Test with numeric data
        numeric_col = [1.0, 2.0, 3.0]
        @test FormulaCompiler.is_mixture_column(numeric_col) == false
        
        # Test with empty column
        empty_col = String[]
        @test FormulaCompiler.is_mixture_column(empty_col) == false
        
        # Test with mixed data (not all elements are mixtures) - should return false
        mixed_col = [mixture_obj, "A", mixture_obj]
        @test FormulaCompiler.is_mixture_column(mixed_col) == false
    end
    
    @testset "extract_mixture_spec" begin
        mixture_obj = mock_mix("A" => 0.3, "B" => 0.7)
        spec = FormulaCompiler.extract_mixture_spec(mixture_obj)
        
        @test spec.levels == ["A", "B"]
        @test spec.weights == [0.3, 0.7]
        @test typeof(spec) <: NamedTuple
        @test haskey(spec, :levels)
        @test haskey(spec, :weights)
    end
    
    @testset "Multiple Level Mixtures" begin
        # Test three-level mixture
        three_level_mixture = mock_mix("A" => 0.2, "B" => 0.3, "C" => 0.5)
        three_level_col = [three_level_mixture, three_level_mixture]
        
        @test FormulaCompiler.is_mixture_column(three_level_col) == true
        
        spec = FormulaCompiler.extract_mixture_spec(three_level_mixture)
        @test spec.levels == ["A", "B", "C"]
        @test spec.weights == [0.2, 0.3, 0.5]
        @test isapprox(sum(spec.weights), 1.0, atol=1e-10)
    end
    
    @testset "Edge Cases" begin
        # Single level mixture (degenerate case)
        single_level_mixture = mock_mix("A" => 1.0)
        single_level_col = [single_level_mixture]
        
        @test FormulaCompiler.is_mixture_column(single_level_col) == true
        spec = FormulaCompiler.extract_mixture_spec(single_level_mixture)
        @test spec.levels == ["A"]
        @test spec.weights == [1.0]
        
        # Test with symbolic levels (as symbols rather than strings)
        # Note: This test assumes the mock mixture can handle string conversion
        # Real implementation might need to handle Symbol inputs differently
    end
end

@testset "MixtureContrastOp Type" begin
    @test FormulaCompiler.MixtureContrastOp <: FormulaCompiler.AbstractOp
    
    # Test type parameter structure
    contrast_matrix = [1.0 0.0; 0.0 1.0; -1.0 -1.0]  # Dummy 3x2 contrast matrix
    
    # Create a MixtureContrastOp instance
    op = FormulaCompiler.MixtureContrastOp{
        :group,           # Column name
        (1, 2),          # Output positions
        (1, 2),          # Level indices  
        (0.3, 0.7)       # Weights
    }(contrast_matrix)
    
    @test op isa FormulaCompiler.MixtureContrastOp
    @test op.contrast_matrix == contrast_matrix
    @test size(op.contrast_matrix) == (3, 2)
end

@testset "Basic Compilation Integration" begin
    # This tests that the mixture detection integrates properly with the compilation system
    # Note: This is a minimal integration test for Phase 1
    # Full compilation and execution will be tested in later phases
    
    @testset "Data Preparation" begin
        # Create test data with mixture column
        mixture_obj = mock_mix("A" => 0.4, "B" => 0.6)
        
        df = DataFrame(
            x = [1.0, 2.0, 3.0],
            y = [0.1, 0.2, 0.3], 
            group = [mixture_obj, mixture_obj, mixture_obj]
        )
        
        data = Tables.columntable(df)
        
        # Test that is_mixture_column works with the data format
        @test FormulaCompiler.is_mixture_column(data.group) == true
        @test FormulaCompiler.is_mixture_column(data.x) == false
        @test FormulaCompiler.is_mixture_column(data.y) == false
        
        # Test mixture spec extraction
        spec = FormulaCompiler.extract_mixture_spec(data.group[1])
        @test spec.levels == ["A", "B"]
        @test spec.weights == [0.4, 0.6]
    end
end

@testset "Phase 2: Mixture Validation" begin
    @testset "validate_mixture_consistency!" begin
        # Test with valid mixture data
        mixture_obj = mock_mix("A" => 0.3, "B" => 0.7)
        valid_data = (
            x = [1.0, 2.0, 3.0],
            group = [mixture_obj, mixture_obj, mixture_obj]
        )
        
        # Should not throw
        @test_nowarn FormulaCompiler.validate_mixture_consistency!(valid_data)
        
        # Test with inconsistent mixtures - need to create data that bypasses is_mixture_column
        # Since is_mixture_column already checks consistency, we need to directly test validate_mixture_column!
        mixture1 = mock_mix("A" => 0.3, "B" => 0.7)
        mixture2 = mock_mix("A" => 0.5, "B" => 0.5)  # Different weights
        inconsistent_column = [mixture1, mixture2]
        
        # This should fail at the column level
        @test_throws ArgumentError FormulaCompiler.validate_mixture_column!(:group, inconsistent_column)
        
        # Test with non-mixture data (should pass)
        regular_data = (
            x = [1.0, 2.0, 3.0],
            group = ["A", "B", "A"]
        )
        
        @test_nowarn FormulaCompiler.validate_mixture_consistency!(regular_data)
    end
    
    @testset "validate_mixture_column!" begin
        # Test weights that don't sum to 1.0
        bad_weights_mixture = unsafe_mock_mixture(["A", "B"], [0.3, 0.6])  # sum = 0.9
        bad_data = [bad_weights_mixture, bad_weights_mixture]
        
        @test_throws ArgumentError FormulaCompiler.validate_mixture_column!(:group, bad_data)
        
        # Test negative weights
        negative_mixture = unsafe_mock_mixture(["A", "B"], [1.3, -0.3])  # sum = 1.0 but negative weight
        negative_data = [negative_mixture, negative_mixture]
        @test_throws ArgumentError FormulaCompiler.validate_mixture_column!(:negative, negative_data)
        
        # Test duplicate levels  
        duplicate_mixture = unsafe_mock_mixture(["A", "A"], [0.5, 0.5])
        duplicate_data = [duplicate_mixture, duplicate_mixture]
        @test_throws ArgumentError FormulaCompiler.validate_mixture_column!(:duplicate, duplicate_data)
        
        # Test empty column (should pass)
        @test_nowarn FormulaCompiler.validate_mixture_column!(:empty, MockMixture[])
    end
end

@testset "Phase 2: Formula Compilation with Mixtures" begin
    @testset "Direct Formula Compilation" begin
        # Create mock categorical data for testing
        # Note: This is a simplified test since we can't easily create CategoricalTerm objects
        # without going through the full StatsModels machinery
        
        # Test basic data validation during compilation
        mixture_obj = mock_mix("A" => 0.4, "B" => 0.6)
        
        test_data = (
            x = [1.0, 2.0, 3.0],
            y = [0.1, 0.2, 0.3],
            group = [mixture_obj, mixture_obj, mixture_obj]
        )
        
        # The validation should pass for consistent mixture data
        @test_nowarn FormulaCompiler.validate_mixture_consistency!(test_data)
        
        # Test with inconsistent data - this should NOT throw because inconsistent
        # mixture columns are not detected as mixture columns by is_mixture_column
        mixture1 = mock_mix("A" => 0.3, "B" => 0.7)
        mixture2 = mock_mix("A" => 0.6, "B" => 0.4)
        
        bad_data = (
            x = [1.0, 2.0],
            group = [mixture1, mixture2]  # Different mixtures
        )
        
        # This should NOT throw because is_mixture_column(bad_data.group) returns false
        @test_nowarn FormulaCompiler.validate_mixture_consistency!(bad_data)
        
        # Verify that is_mixture_column correctly rejects inconsistent data
        @test FormulaCompiler.is_mixture_column(bad_data.group) == false
    end
    
    @testset "Type Parameter Encoding Verification" begin
        # Test that our MixtureContrastOp encodes parameters correctly
        contrast_matrix = [1.0 0.0; 0.0 1.0; -1.0 -1.0]
        
        # Create instance with specific type parameters
        op = FormulaCompiler.MixtureContrastOp{
            :group,                    # Column
            (1, 2),                   # Output positions  
            (1, 2),                   # Level indices
            (0.3, 0.7)               # Weights
        }(contrast_matrix)
        
        # Verify type parameters are correctly embedded
        @test typeof(op).parameters[1] == :group
        @test typeof(op).parameters[2] == (1, 2)
        @test typeof(op).parameters[3] == (1, 2) 
        @test typeof(op).parameters[4] == (0.3, 0.7)
        
        # Test with different parameters creates different type
        op2 = FormulaCompiler.MixtureContrastOp{
            :category,
            (3, 4, 5),
            (1, 2, 3),
            (0.2, 0.3, 0.5)
        }(contrast_matrix)
        
        @test typeof(op) != typeof(op2)  # Different types due to different parameters
        @test typeof(op2).parameters[1] == :category
        @test typeof(op2).parameters[2] == (3, 4, 5)
        @test typeof(op2).parameters[3] == (1, 2, 3)
        @test typeof(op2).parameters[4] == (0.2, 0.3, 0.5)
    end
end

@testset "Phase 3: Execution Engine" begin
    @testset "MixtureContrastOp Execution" begin
        # Create test contrast matrix (3 levels, 2 contrasts - typical dummy coding)
        contrast_matrix = [
            1.0 0.0;    # Level 1: "A"
            0.0 1.0;    # Level 2: "B"  
            0.0 0.0     # Level 3: "C" (reference level)
        ]
        
        # Test general mixture execution
        general_op = FormulaCompiler.MixtureContrastOp{
            :group,                    # Column
            (1, 2),                   # Output positions
            (1, 2, 3),               # Level indices (A=1, B=2, C=3)
            (0.2, 0.3, 0.5)         # Weights (20% A, 30% B, 50% C)
        }(contrast_matrix)
        
        # Test execution
        scratch = zeros(Float64, 5)
        data = (group = ["dummy"],)  # Not used in mixture execution
        
        # Execute the operation
        FormulaCompiler.execute_op(general_op, scratch, data, 1)
        
        # Verify weighted combination: 0.2*[1,0] + 0.3*[0,1] + 0.5*[0,0] = [0.2, 0.3]
        expected_1 = 0.2 * 1.0 + 0.3 * 0.0 + 0.5 * 0.0  # 0.2
        expected_2 = 0.2 * 0.0 + 0.3 * 1.0 + 0.5 * 0.0  # 0.3
        
        @test scratch[1] ≈ expected_1 atol=1e-10
        @test scratch[2] ≈ expected_2 atol=1e-10
        @test scratch[3] == 0.0  # Unused position should remain zero
        @test scratch[4] == 0.0  # Unused position should remain zero
        @test scratch[5] == 0.0  # Unused position should remain zero
    end
    
    @testset "Binary Mixture Optimization" begin
        # Test the optimized binary mixture path
        contrast_matrix = [
            1.0 0.0;    # Level 1: "A"
            0.0 1.0     # Level 2: "B"
        ]
        
        # Create binary mixture operation (should trigger optimized method)
        binary_op = FormulaCompiler.MixtureContrastOp{
            :group,           # Column
            (3, 4),          # Output positions
            (1, 2),          # Level indices (A=1, B=2)
            (0.4, 0.6)       # Weights (40% A, 60% B)
        }(contrast_matrix)
        
        # Test execution with the optimized path
        scratch = zeros(Float64, 6)
        data = (group = ["dummy"],)
        
        FormulaCompiler.execute_op(binary_op, scratch, data, 1)
        
        # Verify binary weighted combination: 0.4*[1,0] + 0.6*[0,1] = [0.4, 0.6]
        @test scratch[3] ≈ 0.4 atol=1e-10
        @test scratch[4] ≈ 0.6 atol=1e-10
        
        # Check unused positions
        @test scratch[1] == 0.0
        @test scratch[2] == 0.0
        @test scratch[5] == 0.0
        @test scratch[6] == 0.0
    end
    
    @testset "Edge Cases" begin
        # Single level mixture (degenerate case)
        single_contrast = reshape([1.0, 0.0], 2, 1)  # 2 levels, 1 contrast
        
        single_op = FormulaCompiler.MixtureContrastOp{
            :group,
            (1,),            # Single output position
            (1,),            # Single level index
            (1.0,)           # Single weight (100%)
        }(single_contrast)
        
        scratch = zeros(Float64, 3)
        data = (group = ["dummy"],)
        
        FormulaCompiler.execute_op(single_op, scratch, data, 1)
        @test scratch[1] ≈ 1.0 atol=1e-10  # 100% of level 1 contrast
        @test scratch[2] == 0.0
        @test scratch[3] == 0.0
        
        # Multiple contrasts with same mixture
        multi_contrast = [
            1.0 0.0 -0.5;    # Level 1
            0.0 1.0 -0.5     # Level 2  
        ]
        
        multi_op = FormulaCompiler.MixtureContrastOp{
            :group,
            (1, 2, 3),       # Three output positions  
            (1, 2),          # Two levels
            (0.7, 0.3)       # 70% level 1, 30% level 2
        }(multi_contrast)
        
        scratch = zeros(Float64, 4)
        FormulaCompiler.execute_op(multi_op, scratch, data, 1)
        
        # Expected: 0.7*[1,0,-0.5] + 0.3*[0,1,-0.5] = [0.7, 0.3, -0.5]
        @test scratch[1] ≈ 0.7 atol=1e-10
        @test scratch[2] ≈ 0.3 atol=1e-10  
        @test scratch[3] ≈ -0.5 atol=1e-10
        @test scratch[4] == 0.0
    end
    
    @testset "Zero-Allocation Verification" begin
        # Test that mixture execution doesn't allocate
        contrast_matrix = [1.0 0.0; 0.0 1.0; -1.0 -1.0]
        
        test_op = FormulaCompiler.MixtureContrastOp{
            :group,
            (1, 2),
            (1, 3),
            (0.8, 0.2)
        }(contrast_matrix)
        
        scratch = zeros(Float64, 5)
        data = (group = ["test"],)
        
        # Warm up to avoid compilation allocations
        FormulaCompiler.execute_op(test_op, scratch, data, 1)
        FormulaCompiler.execute_op(test_op, scratch, data, 1)
        
        # Test zero allocation
        allocs = @allocated FormulaCompiler.execute_op(test_op, scratch, data, 1)
        @test allocs == 0
    end
    
    @testset "Correctness vs Manual Calculation" begin
        # Test against manually computed weighted contrasts
        # Using Helmert coding example
        helmert_matrix = [
            -1.0 -1.0;    # Level 1
             1.0 -1.0;    # Level 2
             0.0  2.0     # Level 3
        ]
        
        # Mixture: 25% level 1, 25% level 2, 50% level 3
        mixture_op = FormulaCompiler.MixtureContrastOp{
            :category,
            (2, 3),
            (1, 2, 3),
            (0.25, 0.25, 0.5)
        }(helmert_matrix)
        
        scratch = zeros(Float64, 5)
        data = (category = ["test"],)
        
        FormulaCompiler.execute_op(mixture_op, scratch, data, 1)
        
        # Manual calculation:
        # 0.25*[-1,-1] + 0.25*[1,-1] + 0.5*[0,2] = [-0.25+0.25+0, -0.25-0.25+1.0] = [0.0, 0.5]
        manual_1 = 0.25 * (-1.0) + 0.25 * 1.0 + 0.5 * 0.0   # 0.0
        manual_2 = 0.25 * (-1.0) + 0.25 * (-1.0) + 0.5 * 2.0  # 0.5
        
        @test scratch[2] ≈ manual_1 atol=1e-10
        @test scratch[3] ≈ manual_2 atol=1e-10
    end
end

@testset "Phase 4: Data Interface and Validation" begin
    @testset "Helper Functions" begin
        @testset "create_mixture_column" begin
            mixture = mock_mix("A" => 0.4, "B" => 0.6)
            
            # Test basic functionality
            col = FormulaCompiler.create_mixture_column(mixture, 5)
            @test length(col) == 5
            @test all(c === mixture for c in col)
            
            # Test zero rows
            empty_col = FormulaCompiler.create_mixture_column(mixture, 0)
            @test length(empty_col) == 0
            
            # Test error for negative rows
            @test_throws ArgumentError FormulaCompiler.create_mixture_column(mixture, -1)
        end
        
        @testset "validate_mixture_weights" begin
            # Valid weights
            @test_nowarn FormulaCompiler.validate_mixture_weights([0.3, 0.7])
            @test_nowarn FormulaCompiler.validate_mixture_weights([0.25, 0.25, 0.5])
            @test_nowarn FormulaCompiler.validate_mixture_weights([1.0])
            
            # Invalid weights - don't sum to 1
            @test_throws ArgumentError FormulaCompiler.validate_mixture_weights([0.3, 0.6])
            @test_throws ArgumentError FormulaCompiler.validate_mixture_weights([0.4, 0.4, 0.4])
            
            # Invalid weights - negative
            @test_throws ArgumentError FormulaCompiler.validate_mixture_weights([0.5, -0.5])
            @test_throws ArgumentError FormulaCompiler.validate_mixture_weights([-0.1, 1.1])
            
            # Test tolerance
            slightly_off = [0.33333333, 0.66666667]  # Sum = 1.00000000
            @test_nowarn FormulaCompiler.validate_mixture_weights(slightly_off, atol=1e-6)
            # Note: The sum might be exactly 1.0 due to floating point, so let's use a more obvious case
            obvious_off = [0.33, 0.66]  # Sum = 0.99
            @test_throws ArgumentError FormulaCompiler.validate_mixture_weights(obvious_off, atol=1e-12)
        end
        
        @testset "validate_mixture_levels" begin
            # Valid levels
            @test_nowarn FormulaCompiler.validate_mixture_levels(["A", "B"])
            @test_nowarn FormulaCompiler.validate_mixture_levels(["Control", "Treatment", "Placebo"])
            @test_nowarn FormulaCompiler.validate_mixture_levels([1, 2, 3])
            
            # Invalid levels - empty
            @test_throws ArgumentError FormulaCompiler.validate_mixture_levels(String[])
            @test_throws ArgumentError FormulaCompiler.validate_mixture_levels(Int[])
            
            # Invalid levels - duplicates
            @test_throws ArgumentError FormulaCompiler.validate_mixture_levels(["A", "A", "B"])
            @test_throws ArgumentError FormulaCompiler.validate_mixture_levels(["X", "Y", "X", "Z"])
        end
        
        @testset "create_balanced_mixture" begin
            # Test binary mixture
            binary = FormulaCompiler.create_balanced_mixture(["A", "B"])
            @test binary["A"] ≈ 0.5
            @test binary["B"] ≈ 0.5
            
            # Test three-way mixture
            three_way = FormulaCompiler.create_balanced_mixture(["X", "Y", "Z"])
            @test three_way["X"] ≈ 1/3 atol=1e-10
            @test three_way["Y"] ≈ 1/3 atol=1e-10
            @test three_way["Z"] ≈ 1/3 atol=1e-10
            
            # Test with different types
            numeric = FormulaCompiler.create_balanced_mixture([1, 2, 3, 4])
            @test numeric["1"] ≈ 0.25
            @test numeric["2"] ≈ 0.25
            @test numeric["3"] ≈ 0.25
            @test numeric["4"] ≈ 0.25
            
            # Test error cases
            @test_throws ArgumentError FormulaCompiler.create_balanced_mixture(String[])
            @test_throws ArgumentError FormulaCompiler.create_balanced_mixture(["A", "A", "B"])  # Duplicates
        end
    end
    
    @testset "expand_mixture_grid" begin
        @testset "Basic Functionality" begin
            # Test with simple base data
            base_data = (x = [1.0, 2.0], y = [0.1, 0.2])
            mixture_specs = Dict(:group => mock_mix("A" => 0.3, "B" => 0.7))
            
            expanded = FormulaCompiler.expand_mixture_grid(base_data, mixture_specs)
            
            @test length(expanded) == 1  # Single combination
            result = expanded[1]
            
            @test haskey(result, :x)
            @test haskey(result, :y) 
            @test haskey(result, :group)
            
            @test result.x == [1.0, 2.0]
            @test result.y == [0.1, 0.2]
            @test length(result.group) == 2
            @test FormulaCompiler.is_mixture_column(result.group)
        end
        
        @testset "Multiple Mixtures" begin
            base_data = (x = [1.0, 2.0, 3.0],)
            mixture_specs = Dict(
                :group => mock_mix("A" => 0.5, "B" => 0.5),
                :treatment => mock_mix("Control" => 0.4, "Treatment" => 0.6)
            )
            
            expanded = FormulaCompiler.expand_mixture_grid(base_data, mixture_specs)
            result = expanded[1]
            
            @test haskey(result, :x)
            @test haskey(result, :group)
            @test haskey(result, :treatment)
            
            @test length(result.x) == 3
            @test length(result.group) == 3
            @test length(result.treatment) == 3
            
            @test FormulaCompiler.is_mixture_column(result.group)
            @test FormulaCompiler.is_mixture_column(result.treatment)
        end
        
        @testset "Edge Cases" begin
            # Empty mixture specs
            base_data = (x = [1.0, 2.0],)
            expanded = FormulaCompiler.expand_mixture_grid(base_data, Dict{Symbol, Any}())
            
            @test length(expanded) == 1
            @test expanded[1] === base_data  # Should return original data
            
            # Override existing column (should warn) - capture the warning
            base_data = (x = [1.0, 2.0], group = ["existing", "data"])
            mixture_specs = Dict(:group => mock_mix("New" => 1.0))
            
            # This should warn but not throw - use @test_logs to capture warning
            @test_logs (:warn, r"Overriding existing column") FormulaCompiler.expand_mixture_grid(base_data, mixture_specs)
            
            # Non-NamedTuple input should error - provide mixtures so we don't hit empty case
            @test_throws ArgumentError FormulaCompiler.expand_mixture_grid(Dict(:x => [1, 2, 3]), Dict(:group => mock_mix("A" => 1.0)))
        end
    end
    
    @testset "Enhanced Validation Error Messages" begin
        # Test that our error messages are clear and actionable
        
        @testset "Weight Validation Messages" begin
            try
                FormulaCompiler.validate_mixture_weights([0.3, 0.6])  # Sum = 0.9
                @test false  # Should not reach here
            catch e
                @test e isa ArgumentError
                @test contains(string(e), "sum to 1.0")
                # The actual sum might be slightly off due to floating point
                @test contains(string(e), "0.8999999") || contains(string(e), "0.9")
            end
            
            try
                FormulaCompiler.validate_mixture_weights([0.5, -0.5])
                @test false  # Should not reach here  
            catch e
                @test e isa ArgumentError
                @test contains(string(e), "non-negative")
            end
        end
        
        @testset "Level Validation Messages" begin
            try
                FormulaCompiler.validate_mixture_levels(String[])
                @test false  # Should not reach here
            catch e
                @test e isa ArgumentError
                @test contains(string(e), "cannot be empty")
            end
            
            try
                FormulaCompiler.validate_mixture_levels(["A", "A", "B"])
                @test false  # Should not reach here
            catch e
                @test e isa ArgumentError
                @test contains(string(e), "unique")
                @test contains(string(e), "A")  # Should identify the duplicate
            end
        end
        
        @testset "Mixture Column Validation Messages" begin
            # Test our existing validation with bad mixtures
            bad_mixture1 = unsafe_mock_mixture(["A", "B"], [0.3, 0.6])  # Bad sum
            bad_mixture2 = unsafe_mock_mixture(["A", "A"], [0.5, 0.5])  # Duplicate levels
            
            try
                FormulaCompiler.validate_mixture_column!(:test_col, [bad_mixture1, bad_mixture1])
                @test false
            catch e
                @test e isa ArgumentError
                @test contains(string(e), "test_col")
                @test contains(string(e), "sum to 1.0")
            end
            
            try
                FormulaCompiler.validate_mixture_column!(:dup_col, [bad_mixture2, bad_mixture2])
                @test false
            catch e
                @test e isa ArgumentError
                @test contains(string(e), "dup_col")
                @test contains(string(e), "duplicate")
            end
        end
    end
    
    @testset "Integration with Existing Systems" begin
        # Test that Phase 4 utilities work with existing compilation system
        
        @testset "Mixture Column Creation + Compilation" begin
            # Create mixture data using Phase 4 utilities
            mixture = mock_mix("A" => 0.3, "B" => 0.7)
            mixture_col = FormulaCompiler.create_mixture_column(mixture, 3)
            
            test_data = (
                x = [1.0, 2.0, 3.0],
                group = mixture_col
            )
            
            # Should pass validation
            @test_nowarn FormulaCompiler.validate_mixture_consistency!(test_data)
            
            # Should be detected as mixture column
            @test FormulaCompiler.is_mixture_column(test_data.group)
            
            # Should extract correct spec
            spec = FormulaCompiler.extract_mixture_spec(test_data.group[1])
            @test spec.levels == ["A", "B"]
            @test spec.weights == [0.3, 0.7]
        end
        
        @testset "Grid Expansion + Validation" begin
            base_data = (x = [1.0, 2.0],)
            mixture_specs = Dict(:group => mock_mix("X" => 0.6, "Y" => 0.4))
            
            expanded = FormulaCompiler.expand_mixture_grid(base_data, mixture_specs)
            expanded_data = expanded[1]
            
            # Should pass all validations
            @test_nowarn FormulaCompiler.validate_mixture_consistency!(expanded_data)
            @test FormulaCompiler.is_mixture_column(expanded_data.group)
            
            # Should work with mixture detection
            spec = FormulaCompiler.extract_mixture_spec(expanded_data.group[1])
            @test spec.levels == ["X", "Y"]
            @test spec.weights == [0.6, 0.4]
        end
        
        @testset "Balanced Mixture Integration" begin
            # Create balanced mixture and use it
            balanced_dict = FormulaCompiler.create_balanced_mixture(["P", "Q", "R"])
            
            # Verify it creates valid weights
            FormulaCompiler.validate_mixture_weights(collect(values(balanced_dict)))
            FormulaCompiler.validate_mixture_levels(collect(keys(balanced_dict)))
            
            # Test weights are actually balanced
            weights = collect(values(balanced_dict))
            @test all(w ≈ 1/3 for w in weights)
        end
    end
end

@debug "Phase 1-4 mixture detection tests completed successfully"