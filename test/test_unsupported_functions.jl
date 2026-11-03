# test_unsupported_functions.jl
# A formula that calls a function the operations cannot evaluate must fail when it is
# compiled, with an ArgumentError naming the term and listing the supported functions.
# Before 1.5.0 an unknown function raised a bare `error("Unknown function: f")`, and a
# supported function with an unsupported argument count (`log(2, z)`) compiled and then
# threw a MethodError at the first evaluation.

using Test
using DataFrames, Tables, GLM, StatsModels, Random
using FormulaCompiler

_cube_plus_one(v) = v^3 + 1    # a user-defined transform

@testset "Unsupported formula functions" begin
    Random.seed!(7)
    n = 40
    df = DataFrame(y = randn(n), x = randn(n), z = rand(n) .+ 0.5, w = rand(n) .+ 0.5)
    data = Tables.columntable(df)

    unsupported = [
        (@formula(y ~ x + _cube_plus_one(z)), "_cube_plus_one(z)", "_cube_plus_one"),
        (@formula(y ~ x + exp2(z)),           "exp2(z)",           "exp2"),
        (@formula(y ~ x + cbrt(z)),           "cbrt(z)",           "cbrt"),
        (@formula(y ~ x + max(z, w)),         "max(z, w)",         "max"),
        (@formula(y ~ x + atan(z, w)),        "atan(z, w)",        "atan"),
        (@formula(y ~ x + log(2, z)),         "log(2, z)",         "2 argument"),  # wrong arity
        (@formula(y ~ x + z & cbrt(w)),       "cbrt(w)",           "cbrt"),        # inside an interaction
        (@formula(y ~ x + log(cbrt(z))),      "cbrt(z)",           "cbrt"),        # nested
    ]
    for (f, term_str, what) in unsupported
        model = lm(f, df)     # StatsModels itself evaluates these fine
        err = try
            compile_formula(model, data)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("`$term_str`", msg)
        @test occursin(what, msg)
        @test occursin("Supported functions: exp, log, log1p, sqrt, abs, sin, cos, inv", msg)
        @test occursin("<, <=, >, >=, ==, !=", msg)
        # The cached entry point fails the same way
        @test_throws ArgumentError get_or_compile_formula(model, data)
    end

    # Every supported function still compiles and reproduces the model matrix
    supported = [
        @formula(y ~ x + exp(z) + log(z) + log1p(z) + sqrt(z) + abs(x) + sin(z) + cos(z) + inv(w)),
        @formula(y ~ x + z^2 + z^0.5 + (z / w) + (z * w) + (x + z + w) + (-(z))),
        @formula(y ~ x + (z > 0.7) + (z <= w) + !(x >= 0) + (x == 0) + (z != 1) + (z < 1)),
        @formula(y ~ x * log(z) + exp(log(abs(x)))),
    ]
    for f in supported
        model = lm(f, df)
        compiled = compile_formula(model, data)
        X = modelmatrix(model)
        out = Vector{Float64}(undef, length(compiled))
        maxdiff = 0.0
        for i in 1:n
            compiled(out, data, i)
            maxdiff = max(maxdiff, maximum(abs.(out .- X[i, :])))
        end
        @test maxdiff == 0.0
    end
end
