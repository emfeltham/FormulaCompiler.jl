# test_execution_plans.jl
# Tests for Phase 1: Core Data Structures

using Test
using FormulaCompiler
using DataFrames, GLM, CategoricalArrays, Tables

using FormulaCompiler:
    ScratchLayout, ExecutionPlan,
    InteractionLayout,
    constant_assignment, data_source,
    ConstantEvaluator, ContinuousEvaluator,
    validate_scratch_layout, validate_execution_plan,
    continuous_assignment, 
    DataSource, ConstantSource, ScratchSource,
    scratch_source, constant_source,
    add_scratch_position!, get_scratch_position,
    output_position, scratch_position,
    add_interaction_layout!, 
    SimpleAssignmentBlock, FunctionBlock,
    function_op, FunctionOp,
    get_evaluator_hash, 
    OutputPosition, ScratchPosition


@testset "Phase 1: Execution Plan Data Structures" begin
    
    @testset "Basic Data Structure Construction" begin
        
        @testset "ScratchLayout" begin
            layout = ScratchLayout(100)
            @test layout.total_size == 100
            @test length(layout.evaluator_positions) == 0
            @test length(layout.interaction_layouts) == 0
            @test validate_scratch_layout(layout)
        end
        
        @testset "ExecutionPlan" begin
            plan = ExecutionPlan(50, 10)
            @test plan.scratch_size == 50
            @test plan.total_output_width == 10
            @test length(plan.blocks) == 0
            @test validate_execution_plan(plan)
        end
        
        @testset "SimpleAssignment" begin
            # Test constant assignment
            const_assign = constant_assignment(5.0, 1)
            @test const_assign.type == :constant
            @test const_assign.source == 5.0
            @test const_assign.output_position == 1
            
            # Test continuous assignment
            cont_assign = continuous_assignment(:x, 2)
            @test cont_assign.type == :continuous
            @test cont_assign.source == :x
            @test cont_assign.output_position == 2
        end
        
        @testset "Input/Output Sources" begin
            # Test input sources
            data_src = data_source(:x)
            @test data_src isa DataSource
            @test data_src.column == :x
            
            scratch_src = scratch_source(10)
            @test scratch_src isa ScratchSource
            @test scratch_src.position == 10
            
            const_src = constant_source(3.14)
            @test const_src isa ConstantSource
            @test const_src.value == 3.14
            
            # Test output destinations
            out_pos = output_position(5)
            @test out_pos isa OutputPosition
            @test out_pos.position == 5
            
            scratch_pos = scratch_position(15)
            @test scratch_pos isa ScratchPosition
            @test scratch_pos.position == 15
        end
    end
    
    @testset "Execution Block Construction" begin
        
        @testset "SimpleAssignmentBlock" begin
            assignments = [
                constant_assignment(1.0, 1),
                continuous_assignment(:x, 2),
                continuous_assignment(:y, 3)
            ]
            
            block = SimpleAssignmentBlock(assignments)
            @test length(block.assignments) == 3
            @test block.assignments[1].type == :constant
            @test block.assignments[2].type == :continuous
            @test block.assignments[3].type == :continuous
        end
        
        @testset "FunctionOp Construction" begin
            # Test function operation: log(x)
            inputs = [data_source(:x)]
            output = output_position(1)
            op = function_op(log, inputs, output)
            
            @test op.func === log
            @test length(op.input_sources) == 1
            @test op.input_sources[1] isa DataSource
            @test op.output_destination isa OutputPosition
        end
    end
    
    @testset "Interaction Layout" begin
        
        @testset "Binary Interaction" begin
            # Test 2×3 interaction (2 components with widths [2, 3])
            component_widths = [2, 3]
            component_positions = [1:2, 3:5]
            output_positions = 1:6
            
            layout = InteractionLayout(
                hash(:test), 
                component_positions, 
                output_positions, 
                component_widths
            )
            
            @test layout.component_widths == [2, 3]
            @test length(layout.kronecker_pattern) == 6  # 2×3 = 6
            
            # Check pattern structure for binary interaction
            pattern = layout.kronecker_pattern
            @test pattern[1] == (1, 1, 0)  # First component: comp1[1] × comp2[1]
            @test pattern[2] == (2, 1, 0)  # Second component: comp1[2] × comp2[1]
            @test pattern[3] == (1, 2, 0)  # Third component: comp1[1] × comp2[2]
            @test pattern[6] == (2, 3, 0)  # Last component: comp1[2] × comp2[3]
        end
        
        @testset "Three-way Interaction" begin
            # Test 2×2×2 interaction
            component_widths = [2, 2, 2]
            component_positions = [1:2, 3:4, 5:6]
            output_positions = 1:8
            
            layout = InteractionLayout(
                hash(:test3), 
                component_positions, 
                output_positions, 
                component_widths
            )
            
            @test layout.component_widths == [2, 2, 2]
            @test length(layout.kronecker_pattern) == 8  # 2×2×2 = 8
            
            # Check first and last pattern elements
            pattern = layout.kronecker_pattern
            @test pattern[1] == (1, 1, 1)  # comp1[1] × comp2[1] × comp3[1]
            @test pattern[8] == (2, 2, 2)  # comp1[2] × comp2[2] × comp3[2]
        end
    end
    
    @testset "Layout Management" begin
        
        @testset "Adding Scratch Positions" begin
            layout = ScratchLayout(100)
            
            # Create dummy evaluators
            const_eval = ConstantEvaluator(5.0)
            cont_eval = ContinuousEvaluator(:x)
            
            # Add scratch positions
            add_scratch_position!(layout, const_eval, 1:5)
            add_scratch_position!(layout, cont_eval, 6:10)
            
            @test length(layout.evaluator_positions) == 2
            @test get_scratch_position(layout, const_eval) == 1:5
            @test get_scratch_position(layout, cont_eval) == 6:10
            
            # Test nonexistent evaluator
            other_eval = ConstantEvaluator(10.0)
            @test get_scratch_position(layout, other_eval) === nothing
        end
        
        @testset "Adding Interaction Layouts" begin
            layout = ScratchLayout(100)
            
            int_layout = InteractionLayout(
                hash(:test),
                [1:2, 3:3],
                1:2,
                [2, 1]
            )
            
            add_interaction_layout!(layout, int_layout)
            @test length(layout.interaction_layouts) == 1
            @test layout.interaction_layouts[1] === int_layout
        end
    end
    
    @testset "ExecutionPlan Management" begin
        
        @testset "Adding Blocks to Plan" begin
            plan = ExecutionPlan(50, 10)
            
            # Add simple assignment block
            assignments = [constant_assignment(1.0, 1)]
            simple_block = SimpleAssignmentBlock(assignments)
            push!(plan, simple_block)
            
            @test length(plan) == 1
            @test plan.blocks[1] === simple_block
            
            # Add function block
            func_block = FunctionBlock(FunctionOp[], UnitRange{Int}[], Int[])
            push!(plan, func_block)
            
            @test length(plan) == 2
            @test plan.blocks[2] === func_block
        end
    end
    
    @testset "Integration with Existing Evaluators" begin
        
        @testset "Evaluator Hash Generation" begin
            # Test that different evaluators get different hashes
            eval1 = ConstantEvaluator(1.0)
            eval2 = ConstantEvaluator(2.0)
            eval3 = ContinuousEvaluator(:x)
            
            hash1 = get_evaluator_hash(eval1)
            hash2 = get_evaluator_hash(eval2)
            hash3 = get_evaluator_hash(eval3)
            
            @test hash1 != hash2  # Different constants
            @test hash1 != hash3  # Different types
            @test hash2 != hash3  # Different types
            
            # Test same evaluator gives same hash
            @test get_evaluator_hash(eval1) == hash1
        end
        
        @testset "Real Evaluator Tree Compatibility" begin
            # Test with actual compiled formula
            df = DataFrame(
                x = [1.0, 2.0, 3.0],
                y = [2.0, 4.0, 6.0],
                group = categorical(["A", "B", "A"])
            )
            
            model = lm(@formula(y ~ x), df)
            compiled = compile_formula(model)
            
            # Test that we can create layouts for real evaluators
            layout = ScratchLayout(50)
            
            # Should be able to add positions for real evaluators
            evaluator = compiled.root_evaluator
            add_scratch_position!(layout, evaluator, 1:10)
            
            retrieved_pos = get_scratch_position(layout, evaluator)
            @test retrieved_pos == 1:10
        end
    end
    
    @testset "Display and Validation" begin
        
        @testset "Pretty Printing" begin
            # Test that show methods don't error
            plan = ExecutionPlan(50, 10)
            layout = ScratchLayout(100)
            
            # Should not error
            @test_nowarn show(IOBuffer(), plan)
            @test_nowarn show(IOBuffer(), layout)
            
            # Test with content
            assignments = [constant_assignment(1.0, 1)]
            block = SimpleAssignmentBlock(assignments)
            push!(plan, block)
            
            @test_nowarn show(IOBuffer(), plan)
            @test_nowarn show(IOBuffer(), block)
        end
        
        @testset "Validation" begin
            # Test valid plans/layouts
            good_plan = ExecutionPlan(50, 10)
            good_layout = ScratchLayout(100)
            
            @test validate_execution_plan(good_plan)
            @test validate_scratch_layout(good_layout)
            
            # Test invalid cases (basic validation)
            bad_plan = ExecutionPlan(-1, 0)  # Negative scratch, zero output
            bad_layout = ScratchLayout(-10)   # Negative size
            
            @test !validate_execution_plan(bad_plan)
            @test !validate_scratch_layout(bad_layout)
        end
    end
end
