# Test AD allocation scaling with formula size

using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays, BenchmarkTools

println("Testing AD allocation scaling with formula size:")
println("=" ^ 60)

for n_cat in [2, 4, 8, 16, 32, 64]
    df = DataFrame(
        y = randn(300),
        x1 = randn(300),
        x2 = randn(300),
        cat = categorical(rand([string(Char(65+i)) for i in 0:(n_cat-1)], 300))
    )

    model = lm(@formula(y ~ x1 + x2 + x1&cat + x2&cat), df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)

    de_ad = derivativeevaluator(:ad, compiled, data, [:x1, :x2])
    J = Matrix{Float64}(undef, length(compiled), 2)

    # Warmup
    derivative_modelrow!(J, de_ad, 1)

    # Measure
    b = @benchmark derivative_modelrow!($J, $de_ad, 1) samples=200

    min_allocs = minimum(b).allocs
    min_bytes = minimum(b).memory

    println("$(rpad(length(compiled), 4)) terms: $(rpad(min_allocs, 3)) allocs, $(rpad(min_bytes, 5)) bytes")
end
