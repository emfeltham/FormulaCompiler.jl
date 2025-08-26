using FormulaCompiler: test_override_compatibility
test_override_compatibility()

## Phase 1

# Test 1: Simple continuous override
df = DataFrame(x = randn(100), y = randn(100))
model = lm(@formula(y ~ x), df)
data = Tables.columntable(df)
scenario = create_scenario("test", data; x = 2.0)
compiled = compile_formula(model, scenario.data)  # Should this work?

# Test 2: Categorical override with new schema system
df = DataFrame(x = randn(100), group = categorical(["A", "B", "C"][rand(1:3, 100)]), y = randn(100))
model = lm(@formula(y ~ x + group), df)
data = Tables.columntable(df)
scenario = create_scenario("test", data; group = "B")
compiled = compile_formula(model, scenario.data)  # Level codes issue?

## Phase 2

# Test 3: Mixed interaction with override
model = lm(@formula(y ~ x * group), df)
scenario = create_scenario("test", data; x = 2.0, group = "A")
compiled = compile_formula(model, scenario.data)  # Schema contrast selection?

## Phase 3

# Test 4: Modelrow with scenarios
row_vec = Vector{Float64}(undef, 10) .* 0.0
modelrow!(row_vec, model, scenario.data, 1)  # Cache key issue?
@assert row_vec == modelmatrix(model)[1, :]

# Test 5: Batch evaluation with scenario collection
collection = create_scenario_grid("test", data, Dict(:x => [1.0, 2.0, 3.0]))
matrix = Matrix{Float64}(undef, 3, 10) .* 0.0
for (i, scenario) in enumerate(collection)
    modelrow!(view(matrix[i, :]), model, scenario.data, 1)
end
