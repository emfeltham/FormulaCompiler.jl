# Profile AD allocation sources for Phase 0 baseline
# This script creates precise measurements of where allocations occur in the AD backend

using FormulaCompiler
using GLM
using DataFrames
using Tables
using BenchmarkTools
using ForwardDiff
using Printf

println("="^80)
println("Phase 0: AD Allocation Profiling Baseline")
println("="^80)
println()

# Helper to format benchmark results
function report_benchmark(name, b)
    @printf("%-50s", name)
    if b.allocs == 0
        @printf(" ✓ ZERO ALLOCS")
    else
        @printf(" ✗ %d allocs", b.allocs)
    end
    @printf(" | %6d bytes | %8.1f ns\n", b.memory, b.time)
end

results = []

println("\n" * "="^80)
println("1. Simple Continuous Variables: y ~ x + z")
println("="^80)

df1 = DataFrame(y = randn(1000), x = randn(1000), z = randn(1000))
model1 = lm(@formula(y ~ x + z), df1)
data1 = Tables.columntable(df1)
compiled1 = compile_formula(model1, data1)
de1_ad = build_derivative_evaluator(compiled1, data1; vars=[:x, :z], backend=:ad)
de1_fd = build_derivative_evaluator(compiled1, data1; vars=[:x, :z], backend=:fd)
g1 = Vector{Float64}(undef, 2)

# Warmup
derivative_modelrow!(g1, de1_ad, coef(model1), 1)
derivative_modelrow!(g1, de1_fd, coef(model1), 1)

b1_ad = @benchmark derivative_modelrow!($g1, $de1_ad, $(coef(model1)), 1) samples=10000 evals=10
b1_fd = @benchmark derivative_modelrow!($g1, $de1_fd, $(coef(model1)), 1) samples=10000 evals=10
report_benchmark("  derivative_modelrow! (AD)", minimum(b1_ad))
report_benchmark("  derivative_modelrow! (FD)", minimum(b1_fd))
push!(results, ("simple_continuous_ad", minimum(b1_ad)))
push!(results, ("simple_continuous_fd", minimum(b1_fd)))

b1_me_ad = @benchmark marginal_effects_eta!($g1, $de1_ad, $(coef(model1)), 1; backend=:ad) samples=10000 evals=10
b1_me_fd = @benchmark marginal_effects_eta!($g1, $de1_fd, $(coef(model1)), 1; backend=:fd) samples=10000 evals=10
report_benchmark("  marginal_effects_eta! (AD)", minimum(b1_me_ad))
report_benchmark("  marginal_effects_eta! (FD)", minimum(b1_me_fd))
push!(results, ("simple_me_eta_ad", minimum(b1_me_ad)))
push!(results, ("simple_me_eta_fd", minimum(b1_me_fd)))

println("\n" * "="^80)
println("2. Interaction Terms: y ~ x * group")
println("="^80)

df2 = DataFrame(y = randn(1000), x = randn(1000), group = rand(["A", "B", "C"], 1000))
model2 = lm(@formula(y ~ x * group), df2)
data2 = Tables.columntable(df2)
compiled2 = compile_formula(model2, data2)
de2_ad = build_derivative_evaluator(compiled2, data2; vars=[:x], backend=:ad)
de2_fd = build_derivative_evaluator(compiled2, data2; vars=[:x], backend=:fd)
g2 = Vector{Float64}(undef, 1)

# Warmup
derivative_modelrow!(g2, de2_ad, coef(model2), 1)
derivative_modelrow!(g2, de2_fd, coef(model2), 1)

b2_ad = @benchmark derivative_modelrow!($g2, $de2_ad, $(coef(model2)), 1) samples=10000 evals=10
b2_fd = @benchmark derivative_modelrow!($g2, $de2_fd, $(coef(model2)), 1) samples=10000 evals=10
report_benchmark("  derivative_modelrow! (AD)", minimum(b2_ad))
report_benchmark("  derivative_modelrow! (FD)", minimum(b2_fd))
push!(results, ("interaction_ad", minimum(b2_ad)))
push!(results, ("interaction_fd", minimum(b2_fd)))

b2_me_ad = @benchmark marginal_effects_eta!($g2, $de2_ad, $(coef(model2)), 1; backend=:ad) samples=10000 evals=10
b2_me_fd = @benchmark marginal_effects_eta!($g2, $de2_fd, $(coef(model2)), 1; backend=:fd) samples=10000 evals=10
report_benchmark("  marginal_effects_eta! (AD)", minimum(b2_me_ad))
report_benchmark("  marginal_effects_eta! (FD)", minimum(b2_me_fd))
push!(results, ("interaction_me_eta_ad", minimum(b2_me_ad)))
push!(results, ("interaction_me_eta_fd", minimum(b2_me_fd)))

println("\n" * "="^80)
println("3. Boolean Predicates: y ~ x + (z > 0)")
println("="^80)

df3 = DataFrame(y = randn(1000), x = randn(1000), z = randn(1000))
model3 = lm(@formula(y ~ x + (z > 0)), df3)
data3 = Tables.columntable(df3)
compiled3 = compile_formula(model3, data3)
de3_ad = build_derivative_evaluator(compiled3, data3; vars=[:x], backend=:ad)
de3_fd = build_derivative_evaluator(compiled3, data3; vars=[:x], backend=:fd)
g3 = Vector{Float64}(undef, 1)

# Warmup
derivative_modelrow!(g3, de3_ad, coef(model3), 1)
derivative_modelrow!(g3, de3_fd, coef(model3), 1)

b3_ad = @benchmark derivative_modelrow!($g3, $de3_ad, $(coef(model3)), 1) samples=10000 evals=10
b3_fd = @benchmark derivative_modelrow!($g3, $de3_fd, $(coef(model3)), 1) samples=10000 evals=10
report_benchmark("  derivative_modelrow! (AD)", minimum(b3_ad))
report_benchmark("  derivative_modelrow! (FD)", minimum(b3_fd))
push!(results, ("boolean_predicate_ad", minimum(b3_ad)))
push!(results, ("boolean_predicate_fd", minimum(b3_fd)))

b3_me_ad = @benchmark marginal_effects_eta!($g3, $de3_ad, $(coef(model3)), 1; backend=:ad) samples=10000 evals=10
b3_me_fd = @benchmark marginal_effects_eta!($g3, $de3_fd, $(coef(model3)), 1; backend=:fd) samples=10000 evals=10
report_benchmark("  marginal_effects_eta! (AD)", minimum(b3_me_ad))
report_benchmark("  marginal_effects_eta! (FD)", minimum(b3_me_fd))
push!(results, ("boolean_me_eta_ad", minimum(b3_me_ad)))
push!(results, ("boolean_me_eta_fd", minimum(b3_me_fd)))

println("\n" * "="^80)
println("4. Link Functions (Logit): y ~ x")
println("="^80)

df4 = DataFrame(y = rand([0, 1], 1000), x = randn(1000))
model4 = glm(@formula(y ~ x), df4, Binomial(), LogitLink())
data4 = Tables.columntable(df4)
compiled4 = compile_formula(model4, data4)
de4_ad = build_derivative_evaluator(compiled4, data4; vars=[:x], backend=:ad)
de4_fd = build_derivative_evaluator(compiled4, data4; vars=[:x], backend=:fd)
g4 = Vector{Float64}(undef, 1)

# Warmup
marginal_effects_mu!(g4, de4_ad, coef(model4), model4.rr.d.link, 1; backend=:ad)
marginal_effects_mu!(g4, de4_fd, coef(model4), model4.rr.d.link, 1; backend=:fd)

b4_mu_ad = @benchmark marginal_effects_mu!($g4, $de4_ad, $(coef(model4)), $(model4.rr.d.link), 1; backend=:ad) samples=10000 evals=10
b4_mu_fd = @benchmark marginal_effects_mu!($g4, $de4_fd, $(coef(model4)), $(model4.rr.d.link), 1; backend=:fd) samples=10000 evals=10
report_benchmark("  marginal_effects_mu! (AD)", minimum(b4_mu_ad))
report_benchmark("  marginal_effects_mu! (FD)", minimum(b4_mu_fd))
push!(results, ("logit_me_mu_ad", minimum(b4_mu_ad)))
push!(results, ("logit_me_mu_fd", minimum(b4_mu_fd)))

println("\n" * "="^80)
println("5. Direct ForwardDiff.jacobian! vs Current Implementation")
println("="^80)

# Test with simple case
nvars = 2
input = randn(nvars)
output = Vector{Float64}(undef, 1)
jac = Matrix{Float64}(undef, 1, nvars)

# Simple closure
f_simple!(out, x) = (out[1] = x[1]^2 + 2*x[1]*x[2] + x[2]^2; out)

cfg = ForwardDiff.JacobianConfig(f_simple!, output, input)

# Warmup
ForwardDiff.jacobian!(jac, f_simple!, output, input, cfg)

b5_jacobian = @benchmark ForwardDiff.jacobian!($jac, $f_simple!, $output, $input, $cfg) samples=10000 evals=10
report_benchmark("  ForwardDiff.jacobian! (cached config)", minimum(b5_jacobian))
push!(results, ("forwarddiff_jacobian", minimum(b5_jacobian)))

# Compare to manual dual construction (current approach simulation)
Tag = typeof(ForwardDiff.Tag(f_simple!, Float64))
DualType = ForwardDiff.Dual{Tag, Float64, nvars}
x_duals = Vector{DualType}(undef, nvars)
partials = [ForwardDiff.Partials{nvars, Float64}(ntuple(i -> i == j ? 1.0 : 0.0, nvars)) for j in 1:nvars]

function manual_dual_approach!(jac, x_duals, partials, input, f!)
    out_vec = similar(jac, size(jac, 1))
    for j in 1:length(input)
        # Rebuild dual (allocates!)
        for i in eachindex(x_duals)
            x_duals[i] = DualType(input[i], partials[j])
        end
        f!(out_vec, x_duals)
        jac[:, j] .= ForwardDiff.partials.(out_vec, 1)
    end
    return jac
end

# Warmup
manual_dual_approach!(jac, x_duals, partials, input, f_simple!)

b5_manual = @benchmark manual_dual_approach!($jac, $x_duals, $partials, $input, $f_simple!) samples=10000 evals=10
report_benchmark("  Manual dual construction (current)", minimum(b5_manual))
push!(results, ("manual_dual_construction", minimum(b5_manual)))

println("\n" * "="^80)
println("SUMMARY: Allocation Sources")
println("="^80)

total_ad_allocs = sum(r[2].allocs for r in results if occursin("_ad", r[1]))
total_ad_memory = sum(r[2].memory for r in results if occursin("_ad", r[1]))
total_fd_allocs = sum(r[2].allocs for r in results if occursin("_fd", r[1]))
total_fd_memory = sum(r[2].memory for r in results if occursin("_fd", r[1]))

println("\nAD Backend:")
@printf("  Total allocations: %d\n", total_ad_allocs)
@printf("  Total memory: %d bytes\n", total_ad_memory)
if total_ad_allocs > 0
    println("  ✗ NOT ZERO-ALLOCATION")
else
    println("  ✓ ZERO-ALLOCATION ACHIEVED")
end

println("\nFD Backend:")
@printf("  Total allocations: %d\n", total_fd_allocs)
@printf("  Total memory: %d bytes\n", total_fd_memory)
if total_fd_allocs > 0
    println("  ✗ NOT ZERO-ALLOCATION")
else
    println("  ✓ ZERO-ALLOCATION ACHIEVED")
end

println("\nAllocation Sources Identified:")
ad_allocating = filter(r -> occursin("_ad", r[1]) && r[2].allocs > 0, results)
if !isempty(ad_allocating)
    println("  AD paths with allocations:")
    for (name, b) in ad_allocating
        @printf("    - %-40s: %d allocs, %d bytes\n", name, b.allocs, b.memory)
    end
else
    println("  ✓ No AD allocation sources found")
end

fd_allocating = filter(r -> occursin("_fd", r[1]) && r[2].allocs > 0, results)
if !isempty(fd_allocating)
    println("  FD paths with allocations:")
    for (name, b) in fd_allocating
        @printf("    - %-40s: %d allocs, %d bytes\n", name, b.allocs, b.memory)
    end
else
    println("  ✓ No FD allocation sources found")
end

println("\n" * "="^80)
println("Comparison: ForwardDiff.jacobian! vs Manual Dual Construction")
println("="^80)
jac_result = results[findfirst(r -> r[1] == "forwarddiff_jacobian", results)]
manual_result = results[findfirst(r -> r[1] == "manual_dual_construction", results)]

@printf("ForwardDiff.jacobian!:      %d allocs, %6d bytes, %8.1f ns\n",
        jac_result[2].allocs, jac_result[2].memory, jac_result[2].time)
@printf("Manual dual construction:   %d allocs, %6d bytes, %8.1f ns\n",
        manual_result[2].allocs, manual_result[2].memory, manual_result[2].time)

if manual_result[2].allocs > jac_result[2].allocs
    println("\n✓ ForwardDiff.jacobian! approach eliminates allocations")
    println("  Recommendation: Proceed with Phase 2 refactor")
else
    println("\n⚠ Both approaches have similar allocation profiles")
    println("  Recommendation: Investigate additional allocation sources")
end

println("\n" * "="^80)
println("Phase 0 Complete - Baseline Established")
println("="^80)
