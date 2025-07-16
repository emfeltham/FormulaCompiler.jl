# test/benchmark.jl
# Comprehensive benchmarking script

using FormulaCompiler
using BenchmarkTools
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels, StandardizedPredictors

println("🚀 FormulaCompiler.jl Benchmark Suite")
println("=" ^ 50)

# Create test data
Random.seed!(42)
n = 10_000
df = DataFrame(
    x = randn(n),
    y = randn(n),
    z = abs.(randn(n)) .+ 0.1,
    group = categorical(rand(["A", "B", "C"], n)),
    flag = rand([true, false], n),
    cat2 = categorical(rand(["X", "Y"], n))
)

data = Tables.columntable(df)

# Test cases with varying complexity
test_cases = [
    (@formula(y ~ x), "Simple linear"),
    (@formula(y ~ x + z), "Multiple continuous"),
    (@formula(y ~ x * group), "Continuous × categorical"),
    (@formula(y ~ x * group + log(z)), "Mixed with function"),
    (@formula(y ~ x * group * z), "Three-way interaction"),
    (@formula(y ~ x * group + log(z) + x^2), "Complex mixed"),
    (@formula(y ~ x * group * flag + log(z) + sqrt(abs(x))), "Very complex")
]

println("\n📊 Performance Benchmarks")
println("-" ^ 30)

results = Dict()

for (formula, description) in test_cases
    println("\n🔍 Testing: $description")
    println("   Formula: $formula")
    
    # Fit model and compile
    model = lm(formula, df)
    compiled = compile_formula(model)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Benchmark compilation
    compile_benchmark = @benchmark compile_formula($model)
    
    # Benchmark evaluation
    eval_benchmark = @benchmark $compiled($row_vec, $data, 1)
    
    # Benchmark vs modelmatrix
    mm_benchmark = @benchmark modelmatrix($model)[1, :]
    
    # Store results
    results[description] = (
        formula = formula,
        compiled = compiled,
        compile_time = compile_benchmark,
        eval_time = eval_benchmark,
        modelmatrix_time = mm_benchmark,
        speedup = median(mm_benchmark.times) / median(eval_benchmark.times)
    )
    
    # Print results
    println("   Compilation: $(BenchmarkTools.prettytime(median(compile_benchmark.times)))")
    println("   Evaluation:  $(BenchmarkTools.prettytime(median(eval_benchmark.times)))")
    println("   ModelMatrix: $(BenchmarkTools.prettytime(median(mm_benchmark.times)))")
    println("   Speedup:     $(round(results[description].speedup, digits=1))x")
    println("   Allocations: $(eval_benchmark.allocs)")
end

println("\n📈 Summary Statistics")
println("-" ^ 30)

# Compilation times
compile_times = [median(r.compile_time.times) for r in values(results)]
println("Compilation time range: $(BenchmarkTools.prettytime(minimum(compile_times))) - $(BenchmarkTools.prettytime(maximum(compile_times)))")

# Evaluation times
eval_times = [median(r.eval_time.times) for r in values(results)]
println("Evaluation time range:  $(BenchmarkTools.prettytime(minimum(eval_times))) - $(BenchmarkTools.prettytime(maximum(eval_times)))")

# Speedups
speedups = [r.speedup for r in values(results)]
println("Speedup range:          $(round(minimum(speedups), digits=1))x - $(round(maximum(speedups), digits=1))x")

# Zero allocations check
zero_alloc_count = sum(r.eval_time.allocs == 0 for r in values(results))
println("Zero allocation cases:  $zero_alloc_count / $(length(results))")

println("\n🎯 Stress Test")
println("-" ^ 30)

# Stress test with most complex case
complex_case = results["Very complex"]
compiled = complex_case.compiled
row_vec = Vector{Float64}(undef, length(compiled))

# Test many evaluations
n_stress = 100_000
println("Running $n_stress evaluations...")

stress_time = @elapsed begin
    for i in 1:n_stress
        row_idx = (i-1) % nrow(df) + 1
        compiled(row_vec, data, row_idx)
    end
end

println("Total time: $(round(stress_time, digits=3))s")
println("Per evaluation: $(round(stress_time / n_stress * 1e9, digits=1))ns")
println("Evaluations/sec: $(round(n_stress / stress_time, digits=0))")

println("\n🔬 Memory Usage")
println("-" ^ 30)

# Memory usage for different data sizes
sizes = [1_000, 10_000, 100_000]
model = lm(@formula(y ~ x * group + log(z)), df)
compiled = compile_formula(model)

for size in sizes
    df_size = DataFrame(
        x = randn(size),
        y = randn(size),
        z = abs.(randn(size)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], size))
    )
    
    data_size = Tables.columntable(df_size)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Measure memory usage
    memory_usage = @allocated begin
        for i in 1:min(1000, size)
            compiled(row_vec, data_size, i)
        end
    end
    
    println("Size $size: $(memory_usage) bytes allocated")
end

println("\n✅ Benchmark Complete!")
println("=" ^ 50)

# Return results for further analysis
results
