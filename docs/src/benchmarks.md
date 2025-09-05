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

3) Scenario Overhead
- Build `scenario = create_scenario("policy", data; x=2.0, group = first(levels(df.group)))`
- Compare compiled(row,data,i) vs compiled(row,scenario.data,i)
- Target: identical times within noise; 0 allocations

4) Derivative Jacobian (AD and FD)
- Build evaluator: `vars = continuous_variables(compiled, data)`; `de = build_derivative_evaluator(compiled, data; vars=vars)`
- AD Jacobian: `J = similar(rand(length(compiled), length(vars))); @benchmark derivative_modelrow!($J, $de, 25)`
- FD single-column: `col = similar(rand(length(compiled))); @benchmark fd_jacobian_column!($col, $de, 1, 25)`
- Targets: AD ≤512 bytes; FD 0 bytes

5) Marginal Effects (η and μ)
- `β = coef(model)`; `g = similar(rand(length(vars)))`
- η-scale AD: `@benchmark marginal_effects_eta!($g, $de, $β, 25; backend=:ad)`
- η-scale FD: `@benchmark marginal_effects_eta!($g, $de, $β, 25; backend=:fd)` (or `marginal_effects_eta_fd!`)
- μ-scale with link (e.g., Logit): `@benchmark marginal_effects_mu!($g, $de, $β, 25; link=LogitLink(), backend=:ad)` and `backend=:fd`
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
