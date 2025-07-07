# runtests.jl

using Revise
using EfficientModelMatrices

using Test, StatsModels, DataFrames, CategoricalArrays, GLM

# Include our zero-allocation implementation

# Create test dataset
n = 600_000
df = DataFrame(
    y   = randn(n),
    x1  = randn(n),
    x2  = randn(n),
    g   = categorical(rand(["A","B","C"], n)),
    h   = categorical(rand(Bool, n)),
)

# Create formula and model
f = @formula(y ~ x1 + x2 + g + h + x1 & g + x1 & x2 + g & h)
m1 = lm(f, df)

# Get reference matrix using standard StatsModels
Xref = modelmatrix(m1);
size(Xref)

println("Reference matrix size: ", size(Xref))
println("Reference matrix type: ", typeof(Xref))

# Test our zero-allocation version
X = Matrix{Float64}(undef, size(Xref));
X .= 0.0;

Revise.retry()

# Time the operation
println("\nTiming zero-allocation version:")
# @time modelcols!(X, m1, df);

@time ipm = InplaceModeler(m1, nrow(df));
@time modelmatrix2!(ipm, Tables.columntable(df), X);

using Profile
@profile modelmatrix2!(ipm, Tables.columntable(df), X);

open("profile_output_emm.txt", "w") do io
    Profile.print(io)
end

let h = 8
    @test X[:, h] ≈ Xref[:, h]
end

# Verify correctness
@test X ≈ Xref

println("✓ Matrix values match reference implementation")

# Test with different data to ensure no state leakage
df2 = deepcopy(df)
df2.x1 .+= 1

X2 = Matrix{Float64}(undef, size(Xref))
modelcols!(X2, f.rhs, df2, m1)

# Get reference for modified data
Xref2 = modelmatrix(f.rhs, df2)
@test X2 ≈ Xref2

println("✓ Works correctly with different data")

# Test allocation behavior
println("\nAllocation test:")
println("Baseline (should be minimal):")
@time modelcols!(X, f.rhs, df, m1)

println("\nReference StatsModels (for comparison):")
@time Xref_new = modelmatrix(f.rhs, df)

println("\n✓ All tests passed!")

# Simple benchmark
using BenchmarkTools

println("\nBenchmark comparison:")
println("Zero-allocation version:")
@btime modelcols!($X, $(f.rhs), $df, $m1)

println("Standard StatsModels:")
@btime modelmatrix($(f.rhs), $df)
