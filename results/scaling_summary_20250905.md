 # Scaling Summary (Measured)

 This file consolidates scaling evidence across two benchmark runs.

 ## Environments

 - Env A (size invariance): Julia 1.11.2, Threads 2, CPU apple-m1, OS Darwin; FormulaCompiler 1.0.0; GLM 1.9.0; MixedModels 4.38.1; ForwardDiff 1.1.0
   - Source: results/benchmarks_20250905_143814.md
 - Env B (complex vs simple): Julia 1.11.2, Threads 1, CPU apple-m1, OS Darwin; FormulaCompiler 1.0.0; GLM 1.9.0; MixedModels 4.38.1; ForwardDiff 1.1.0
   - Source: results/benchmarks_20250905_145246_scaling.md

 ## Per‑Row Invariance by Data Size (Env A)

 Complex, interaction‑heavy formula; per‑row evaluation medians remain effectively constant as n increases; allocations 0 B.

 | n (rows) | Median (ns) | Min (ns) | Min Mem (B) |
 |----------|-------------|----------|--------------|
 | 10,000   | 13.6        | 12.1     | 0            |
 | 100,000  | 13.6        | 12.1     | 0            |
 | 1,000,000| 13.7        | 12.1     | 0            |

 ## Complex vs Simple Formula (Env B)

 Both maintain 0 B allocations; per‑row latency remains in single‑digit to low‑teens nanoseconds.

 | Formula | Median (ns) | Min (ns) | Min Mem (B) |
 |---------|-------------|----------|--------------|
 | Simple  | 6.8         | 6.4      | 0            |
 | Complex | 13.6        | 12.1     | 0            |

 ## Notes

 - Per‑row latency and zero‑allocation behavior hold across dataset sizes and complex interaction patterns.
 - Small differences across runs reflect configuration (Threads=1 vs 2) and normal benchmark variance.
