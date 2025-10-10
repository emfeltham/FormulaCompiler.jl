# Benchmark Protocol

Purpose
- Provide a reproducible method to validate performance and allocation claims.
- Report results with context and conservative interpretation.

Environment
- Julia: record `VERSION` (e.g., 1.10.x or 1.11.x)
- CPU: model, frequency, cores; `Sys.CPU_NAME` if available
- OS: name and version
- Threads: `Threads.nthreads()` and BLAS threads (`BLAS.get_num_threads()`)
- Packages: `Pkg.status()` for FormulaCompiler, GLM, MixedModels, ForwardDiff, Tables, DataFrames

Runner Script
- A convenience runner is provided at `scripts/benchmarks.jl`.
- Examples:
  - `julia --project=. scripts/benchmarks.jl` (default subset)
  - `julia --project=. scripts/benchmarks.jl core deriv margins se` (select tasks)
- Prints per-benchmark median/min time and minimum memory; include this output with your Environment section.

Minimal runner
- Edit options at the top of `scripts/benchmarks_simple.jl` and run:
  - `julia --project=. scripts/benchmarks_simple.jl`
- Set `selected`, `fast`, `out`, `file`, `tag` in the script; no CLI flags needed.

Setup
- Activate project: `Pkg.activate("..")`; `Pkg.instantiate()`
- Using: `using BenchmarkTools, FormulaCompiler, GLM, MixedModels, Tables, DataFrames, CategoricalArrays, Random`
- Data format: Prefer `Tables.columntable(df)` for evaluation
- Warmup: run each function once before benchmarking

Conventions
- Use `@benchmark` with concrete arguments; avoid globals
- Record: `minimum(time)`, `median(time)`, `minimum(memory)`
- Target allocations: FD paths 0 bytes; AD paths ≤512 bytes (ForwardDiff overhead)
- Present ranges, not single points; note environment details

Benchmarks

1) Core Row Evaluation (Compiled)
- Model: small GLM (e.g., `y ~ x * group + log1p(abs(z))` with categorical `group`)
- Steps:
  - Fit model with GLM; build `data = Tables.columntable(df)`
  - `compiled = compile_formula(model, data)`
  - `row = Vector{Float64}(undef, length(compiled))`
  - Single row: `@benchmark $compiled($row, $data, 25)`
  - Tight loop (amortized): call inside a loop over indices; check stability
- Targets: ≈O(10^1–10^2) ns; `minimum(memory) == 0`

2) Allocating vs In-place Interfaces
- `modelrow(model, data, i)` vs `modelrow!` with preallocated buffer
- Targets: In-place 0 bytes; allocating shows expected vector/matrix allocations

3) Counterfactual Overhead
- Build counterfactual data: `data_cf, cf_vecs = build_counterfactual_data(data, [:x, :group], 1)`
- Update replacements: `update_counterfactual_replacement!(cf_vecs[1], 2.0)`
- Compare compiled(row,data,i) vs compiled(row,data_cf,i)
- Target: identical times within noise; 0 allocations

4) Derivative Jacobian (AD and FD)
- Build evaluators: `vars = continuous_variables(compiled, data)`; `de_ad = derivativeevaluator_ad(compiled, data, vars)`; `de_fd = derivativeevaluator_fd(compiled, data, vars)`
- AD Jacobian: `J = similar(rand(length(compiled), length(vars))); @benchmark derivative_modelrow!($J, $de_ad, 25)`
- FD Jacobian: `@benchmark derivative_modelrow!($J, $de_fd, 25)`
- FD single-column: `col = similar(rand(length(compiled))); @benchmark fd_jacobian_column!($col, $de_fd, 1, 1)`
- Targets: AD ≤512 bytes; FD 0 bytes

5) Marginal Effects (η and μ)
- `β = coef(model)`; `g = similar(rand(length(vars)))`
- η-scale AD: `@benchmark marginal_effects_eta!($g, $de_ad, $β, 25)`
- η-scale FD: `@benchmark marginal_effects_eta!($g, $de_fd, $β, 25)`
- μ-scale with link (e.g., Logit): `@benchmark marginal_effects_mu!($g, $de_ad, $β, 25, LogitLink())` and `@benchmark marginal_effects_mu!($g, $de_fd, $β, 25, LogitLink())`
- Targets: FD 0 bytes; AD ≤512 bytes

6) Delta Method SE
- `gβ = similar(rand(length(β))); Σ = I*1.0` (or `vcov(model)`)
- `@benchmark delta_method_se($gβ, $Σ)`
- Target: 0 bytes, O(10^1) ns for dense small Σ

7) MixedModels Fixed Effects
- Fit a small LMM/GLMM; compile with fixed-effects extraction
- Benchmark compiled row evaluation as in (1)
- Target: Same guarantees (0 bytes; similar timing)

8) Scaling and Complexity
- Simple vs complex formulas; small vs larger `OutputSize`
- Report time growth and allocation behavior (should remain 0 for core paths)

9) Size Invariance (Per-Row)
- Use `scale_n` to run the same per-row evaluation on increasing data sizes (e.g., 10k, 100k, 1M)
- Expectation: per-row latency and allocations remain effectively constant (0 B) as `n` increases

Reporting Template
- Environment summary
- For each benchmark: code snippet, median/min time, allocations, brief interpretation
- Deviations: note and investigate; check warmup, data format, thread counts

Edge Cases
- Integer columns: verify automatic Float64 conversion in evaluator paths
- Large tuples: ensure hybrid dispatch preserves 0 allocations
- Categorical mixtures: include a case if used (see mixtures guide)

Interpretation
- Treat numbers as indicative; absolute values vary by system
- Prioritize allocation guarantees and scaling trends over exact nanoseconds

# End
