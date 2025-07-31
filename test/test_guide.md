# FormulaCompiler.jl Testing

Cf. "runtests.jl"

## Test core system

```bash
julia --project="." test/test_position_mapping.jl  &> run_debug.
julia --project="." test/test_step1_specialized_core.jl  &> run_debug.out
julia --project="." test/test_step2_categorical_support.jl &> run_debug.out
julia --project="." test/test_step3_function_support.jl &> run_debug.out
julia --project="." test/test_step4_interactions.jl &> run_debug.out
```

```bash
julia --project="." test/test_override_1_2.jl &> run_debug.out
julia --project="." test/test_override_3.jl &> run_debug.out
julia --project="." test/test_override_4.jl &> run_debug.out
```

## Ancillary tests

```bash
julia --project="." step4_run_profiling.jl  &> run_debug.out
```

## `modelrow()` tests

```bash
julia --project="." test/test_modelrow.jl  &> run_debug.out
julia --project="." X  &> run_debug.out
```

## Derivative tests

```bash
julia --project="." test_derivative_phase1.jl  &> run_debug.out
julia --project="." test_derivative_phase2.jl  &> run_debug.out
```
