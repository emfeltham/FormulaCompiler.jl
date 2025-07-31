# example.jl

# Basic test

# User creates model and data
df = DataFrame(
    x = randn(10),
    y = randn(10),
    z = abs.(randn(10)) .+ 0.1,
    group = categorical(rand(["A", "B", "C"], 10))
)
model = fit(LinearModel, @formula(y ~ x * group), df)
data = Tables.columntable(df);

# Compile formula (requires explicit data in v2.0)
compiled = compile_formula(model, data)

# Evaluate rows with zero allocation
output = Vector{Float64}(undef, length(compiled))
out = compiled(output, data, 1)

# Correctness
@test out ≈ eachrow(modelmatrix(model))[1]

###############################################################################
# Test
###############################################################################

function test_execution_blocks()
    df = DataFrame(
        x = [1.0, 2.0, 3.0],
        group = categorical(["A", "B", "A"]),
        y = [4.0, 5.0, 6.0]
    )
    data = Tables.columntable(df)
    
    # Test each formula type
    formulas = [
        @formula(y ~ 1),           # AssignmentBlock (constant)
        @formula(y ~ x),           # AssignmentBlock (continuous)  
        @formula(y ~ group),       # CategoricalBlock
        @formula(y ~ x + group),   # Mixed blocks
        @formula(y ~ x * group)    # InteractionBlock
    ]
    
    for formula in formulas
        model = lm(formula, df)
        compiled = compile_formula(model, data)
        
        # Test execution plan works
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, data, 1)
        
        # Compare with expected
        expected = modelmatrix(model)[1, :]
        @test output ≈ expected
        
        println("✅ $formula: execution plan works")
    end
end

function test_zero_allocation_execution_plans()
    df = DataFrame(x = [1.0, 2.0], y = [3.0, 4.0])
    data = Tables.columntable(df)
    model = lm(@formula(y ~ x), df)
    compiled = compile_formula(model, data)
    
    output = Vector{Float64}(undef, length(compiled))
    
    # Warmup
    compiled(output, data, 1)
    
    # Test allocation
    alloc = @allocated compiled(output, data, 1)
    
    @test alloc == 0
    println("Execution plans: $alloc bytes allocated")
end

test_execution_blocks()
test_zero_allocation_execution_plans()