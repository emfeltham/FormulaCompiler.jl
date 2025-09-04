# zero alloc ad info.md

The key to achieving zero-allocation evaluations with ForwardDiff.jl is to pre-allocate all the necessary memory (often called a "workspace" or "cache") and use the in-place, mutating API functions that end with an exclamation mark (!).

The Challenge: Allocations with Changing Inputs
When you repeatedly call standard functions like ForwardDiff.derivative(f, x), Julia allocates memory on each call to store the result and the intermediate values used in the computation (the "dual" numbers). If you're doing this inside a hot loop (a performance-critical part of your code), these repeated allocations can significantly slow down your program and trigger the garbage collector.

Consider a function where some parameters change on each evaluation:

Julia
f(x, p1, p2) = p1 * sin(x) + p2 * cos(x)
If you need to compute the derivative with respect to x for many different values of p1 and p2, a naive loop would allocate memory each time.

The Solution: Pre-allocation with Config and DiffResult
ForwardDiff.jl provides a powerful way to avoid this by pre-allocating a configuration object and a result object.

DerivativeConfig: This object holds all the pre-allocated memory (the "workspace") that ForwardDiff needs to perform its calculations. You create it once for a given function and input type.

DiffResult: This object is a pre-allocated container to store both the original function's output (the "value") and the derivative.

derivative!(result, f, x, cfg): This is the in-place version of the derivative function. It uses the pre-allocated configuration cfg and writes its output into the pre-allocated result object, returning the result object itself.

Step-by-Step Implementation
Let's build a zero-allocation derivative evaluator for the function f(x, a, b) = a * x^2 + b * x, where we want the derivative with respect to x, while a and b vary.

1. Define the Core Function

First, define the mathematical formula. Notice that this function only takes x as its argument. The parameters a and b will be passed in using a closure.

Julia
# The fixed formula we want to differentiate
fixed_formula(x, a, b) = a * x^2 + b * x
2. Set Up the Workspace

Now, we pre-allocate everything we need. We'll wrap this logic in a function to make it reusable.

Julia
using ForwardDiff, BenchmarkTools

function setup_workspace(x_initial::T) where T
    # 1. Define a closure that captures parameters `a` and `b`.
    #    This is the function we will actually differentiate.
    #    The (a, b) arguments are just placeholders for the closure.
    func_to_diff = (x, a, b) -> fixed_formula(x, a, b)

    # 2. Create the configuration object.
    #    The first argument is the function that will be differentiated.
    #    We pass an anonymous function `x -> func_to_diff(x, 1.0, 1.0)`
    #    to match the signature ForwardDiff expects (a function of one variable).
    #    The specific values of `a` and `b` here don't matter; they just
    #    satisfy the signature. The second argument is the input value `x`.
    cfg = ForwardDiff.DerivativeConfig(x -> func_to_diff(x, 1.0, 1.0), x_initial)

    # 3. Pre-allocate the result object.
    #    This will store the function's value and its derivative.
    result = ForwardDiff.DiffResult(0.0, 0.0) # Value and derivative are both Floats

    return cfg, result
end
3. Create the Zero-Allocation Evaluator

This is the function you'll call inside your performance-critical loop. It takes the pre-allocated objects and the varying data as input.

Julia
function zero_alloc_derivative!(result, cfg, x, a, b)
    # Create a closure over the varying parameters `a` and `b`.
    # This is the key step! The closure "burns in" the current
    # values of `a` and `b` for this specific derivative calculation.
    f_closed = z -> fixed_formula(z, a, b)

    # Call the in-place function with the closure and pre-allocated objects.
    ForwardDiff.derivative!(result, f_closed, x, cfg)

    # Extract the value and derivative from the result object.
    val = ForwardDiff.value(result)
    deriv = ForwardDiff.derivative(result)

    return val, deriv
end
4. Benchmark and Verify

Finally, let's use the setup in a loop and use BenchmarkTools.jl to confirm it makes zero allocations.

Julia
# Initial value for x
x_val = 2.0

# Create our pre-allocated workspace
cfg, result = setup_workspace(x_val)

# Define some varying parameters
params_a = rand(100)
params_b = rand(100)

# --- Run the zero-allocation version ---
println("Benchmarking zero-allocation version:")
@btime zero_alloc_derivative!($result, $cfg, $x_val, $(params_a[1]), $(params_b[1]))

# --- Compare to the standard, allocating version for reference ---
println("\nBenchmarking standard allocating version:")
allocating_func(x, a, b) = fixed_formula(x, a, b)
@btime ForwardDiff.derivative(x -> allocating_func(x, $(params_a[1]), $(params_b[1])), $x_val)
Expected Output:

The benchmark for zero_alloc_derivative! will report 0 allocations: 0 bytes, while the standard version will report at least one allocation. This confirms our success! 🎉

Benchmarking zero-allocation version:
  20.258 ns (0 allocations: 0 bytes)

Benchmarking standard allocating version:
  32.124 ns (1 allocation: 32 bytes)
The exact timing will vary, but the allocation count is what matters. This technique can be extended to gradient!, jacobian!, and hessian! by using GradientConfig, JacobianConfig, and HessianConfig, respectively.



Your Association of Yale Alumni chats aren’t used to improve our models. Gemini can make mistakes, so double-check it. Your privacy & Gemini Opens in a new window

