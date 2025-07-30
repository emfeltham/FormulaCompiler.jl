#!/usr/bin/env julia

using Test
using DataFrames, Tables
using FormulaCompiler
using FormulaCompiler: 
    FunctionEvaluator, ContinuousEvaluator, ConstantEvaluator,
    ValidatedExecutionPlan, analyze_scratch_requirements,
    execute_plan!, output_width, ExecutionPlan, ScratchLayout,
    decompose_evaluator_tree, generate_function_block!, compile_formula,

# Helper to build evaluators for test cases
function build_test_evaluator(description::String, vars)
    if description == "log(x)"
        return FunctionEvaluator(log, [ContinuousEvaluator(:x)])
    elseif description == "x^2"
        return FunctionEvaluator(^, [ContinuousEvaluator(:x), ConstantEvaluator(2.0)])
    elseif description == "x + y"
        return FunctionEvaluator(+, [ContinuousEvaluator(:x), ContinuousEvaluator(:y)])
    elseif description == "log(x^2)"
        x_eval = ContinuousEvaluator(:x)
        two_eval = ConstantEvaluator(2.0)
        power_eval = FunctionEvaluator(^, [x_eval, two_eval])
        return FunctionEvaluator(log, [power_eval])
    elseif description == "sin(x + y)"
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        add_eval = FunctionEvaluator(+, [x_eval, y_eval])
        return FunctionEvaluator(sin, [add_eval])
    elseif description == "exp(log(x))"
        x_eval = ContinuousEvaluator(:x)
        log_eval = FunctionEvaluator(log, [x_eval])
        return FunctionEvaluator(exp, [log_eval])
    elseif description == "sqrt(x^2 + y^2)"
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        x2 = FunctionEvaluator(^, [x_eval, ConstantEvaluator(2.0)])
        y2 = FunctionEvaluator(^, [y_eval, ConstantEvaluator(2.0)])
        sum_eval = FunctionEvaluator(+, [x2, y2])
        return FunctionEvaluator(sqrt, [sum_eval])
    else
        error("Unknown test case: $description")
    end
end

# Execute decomposed operations from AST decomposition
function execute_decomposed_operations(operations, scratch, data, row_idx)
    for op in operations
        args = Float64[]
        for ref in op.input_refs
            if ref.location_type == :data
                push!(args, data[ref.index][row_idx])
            elseif ref.location_type == :scratch
                push!(args, scratch[ref.index])
            elseif ref.location_type == :constant
                push!(args, ref.index)
            end
        end
        result = isempty(op.func) ? args[1] : apply_function_safe(op.func, args...)
        if op.output_ref.location_type == :scratch
            scratch[op.output_ref.index] = result
        end
    end
    return scratch[end]
end

# Prepare sample DataFrame and columnar data
const df = DataFrame(x = [1.0, 2.0, 3.0, 4.0], y = [2.0, 3.0, 4.0, 5.0], z = [0.1, 0.2, 0.3, 0.4])
const data = Tables.columntable(df)

# Test cases: (description, function, vars...)
const test_cases = [
    ("log(x)", x -> log(x), :x),
    ("x^2", x -> x^2, :x),
    ("x + y", (x,y) -> x+y, [:x, :y]),
    ("log(x^2)", x -> log(x^2), :x),
    ("sin(x + y)", (x,y) -> sin(x+y), [:x, :y]),
    ("exp(log(x))", x -> exp(log(x)), :x),
    ("sqrt(x^2 + y^2)", (x,y) -> sqrt(x^2+y^2), [:x, :y])
]

@testset "AST" begin

    @testset "AST Decomposition" begin
        for (desc, f, vars) in test_cases
            ev = build_test_evaluator(desc, vars)
            layout = ScratchLayout()
            ops = decompose_evaluator_tree(ev, layout)
            # Allocate scratch
            maxidx = maximum([r.index for r in getfield.(ops, :output_ref) if r.location_type == :scratch]; init=0)
            scratch = Vector{Float64}(undef, max(maxidx,1))
            result = execute_decomposed_operations(ops, scratch, data, 1)
            expected = isa(vars, Symbol) ? f(df[1, vars]) : f([df[1, v] for v in vars]...)
            @test isapprox(result, expected; atol=1e-12)
        end
    end

    @testset "Execution Plan Integration" begin
        # Build complex function log(x^2)
        x_eval = ContinuousEvaluator(:x)
        two_ev = ConstantEvaluator(2.0)
        pow_ev = FunctionEvaluator(^, [x_eval, two_ev])
        log_ev = FunctionEvaluator(log, [pow_ev])

        layout = analyze_scratch_requirements(log_ev)
        plan = ExecutionPlan(layout.total_size, output_width(log_ev))
        _ = generate_function_block!(plan, log_ev, 1, layout)

        @test length(plan.blocks) > 0
        scratch = Vector{Float64}(undef, max(plan.scratch_size,1))
        output = Vector{Float64}(undef, plan.total_output_width)
        execute_plan!(plan, scratch, output, data, 1)
        @test isapprox(output[1], log(1.0^2); atol=1e-12)
    end

    @testset "Zero Allocation" begin
        ev = build_test_evaluator("log(x^2)", :x)
        vplan = ValidatedExecutionPlan(ev, data)
        scratch = Vector{Float64}(undef, vplan.scratch_size)
        out = Vector{Float64}(undef, vplan.total_output_width)
        # Warmup
        execute_plan!(vplan, scratch, out, data, 1)
        allo = @allocated execute_plan!(vplan, scratch, out, data, 1)
        @test allo <= 64
    end
end