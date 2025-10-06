# Test if ForwardDiff.jacobian! approach would be better

using ForwardDiff, BenchmarkTools

# Simulate our compiled function
function test_func!(output, x_vec)
    output[1] = 1.0           # intercept
    output[2] = x_vec[1]      # x
    output[3] = x_vec[2]      # z
    return nothing
end

# Setup
nvars = 2
nout = 3
x = [1.5, 2.5]
output = zeros(nout)

# Approach 1: Current manual dual propagation
println("=" ^ 70)
println("Approach 1: Manual Dual Propagation (current)")
println("=" ^ 70)

DualT = ForwardDiff.Dual{Nothing, Float64, nvars}
x_dual = Vector{DualT}(undef, nvars)
partials1 = ForwardDiff.Partials{nvars, Float64}((1.0, 0.0))
partials2 = ForwardDiff.Partials{nvars, Float64}((0.0, 1.0))
output_dual = Vector{DualT}(undef, nout)

# Warmup
for _ in 1:100
    x_dual[1] = DualT(x[1], partials1)
    x_dual[2] = DualT(x[2], partials2)
    test_func!(output_dual, x_dual)
end

# Benchmark
b1 = @benchmark begin
    x_dual[1] = DualT(\$x[1], \$partials1)
    x_dual[2] = DualT(\$x[2], \$partials2)
    test_func!(\$output_dual, \$x_dual)
end samples=1000 evals=1

println("Memory: ", minimum(b1).memory, " bytes")
println("Allocs: ", minimum(b1).allocs)
println("Time:   ", round(minimum(b1).time, digits=1), " ns")

# Approach 2: ForwardDiff.jacobian! with closure
println("\n" * "=" ^ 70)
println("Approach 2: ForwardDiff.jacobian! with cached config")
println("=" ^ 70)

# Closure that captures output buffer
function make_closure(output_buf)
    return function(x_vec)
        test_func!(output_buf, x_vec)
        return output_buf
    end
end

output_buf = zeros(nout)
closure = make_closure(output_buf)
J = zeros(nout, nvars)

# Build config
chunk = ForwardDiff.Chunk{nvars}()
cfg = ForwardDiff.JacobianConfig(closure, x, chunk)

# Warmup
for _ in 1:100
    ForwardDiff.jacobian!(J, closure, x, cfg)
end

# Benchmark
b2 = @benchmark ForwardDiff.jacobian!(\$J, \$closure, \$x, \$cfg) samples=1000 evals=1

println("Memory: ", minimum(b2).memory, " bytes")
println("Allocs: ", minimum(b2).allocs)
println("Time:   ", round(minimum(b2).time, digits=1), " ns")

println("\n" * "=" ^ 70)
println("CONCLUSION")
println("=" ^ 70)
println("Manual dual propagation: ", minimum(b1).memory, " bytes")
println("ForwardDiff.jacobian!:   ", minimum(b2).memory, " bytes")

if minimum(b2).memory < minimum(b1).memory
    println("\n✅ ForwardDiff.jacobian! is BETTER (fewer allocations)")
else
    println("\n❌ Manual approach is still better OR jacobian! also allocates")
end
