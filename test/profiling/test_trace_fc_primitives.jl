# Trace directly into FormulaCompiler primitives to find allocation source
using Margins, GLM, DataFrames, Tables, BenchmarkTools, CategoricalArrays
using Margins: build_engine, PopulationUsage, HasDerivatives
using FormulaCompiler: contrast_modelrow!, contrast_gradient!

println("\n" * "="^80)
println("Tracing FormulaCompiler Primitives")
println("Testing contrast_modelrow! and contrast_gradient! directly")
println("="^80)

n = 100

# Boolean categorical
df_bool = DataFrame(y = randn(n), x = randn(n), treated = rand([false, true], n))
model_bool = lm(@formula(y ~ x + treated), df_bool)
data_nt_bool = Tables.columntable(df_bool)
engine_bool = build_engine(PopulationUsage, HasDerivatives, model_bool, data_nt_bool, [:treated], GLM.vcov, :ad)

# String categorical
df_string = DataFrame(y = randn(n), x = randn(n), group = categorical(rand(["Control", "Treatment"], n)))
model_string = lm(@formula(y ~ x + group), df_string)
data_nt_string = Tables.columntable(df_string)
engine_string = build_engine(PopulationUsage, HasDerivatives, model_string, data_nt_string, [:group], GLM.vcov, :ad)

println("\n1. Test contrast_modelrow! directly")
println("-"^80)

# Boolean
buf_bool = Vector{Float64}(undef, length(coef(model_bool)))
for _ in 1:10
    contrast_modelrow!(buf_bool, engine_bool.contrast, 1, :treated, false, true)
end

b_bool_modelrow = @benchmark contrast_modelrow!(
    $buf_bool, $(engine_bool.contrast), 1, :treated, false, true
) samples=200

println("\nBoolean - contrast_modelrow!:")
println("  Allocations: ", b_bool_modelrow.allocs)
println("  Memory:      ", b_bool_modelrow.memory, " bytes")

# String
buf_string = Vector{Float64}(undef, length(coef(model_string)))
for _ in 1:10
    contrast_modelrow!(buf_string, engine_string.contrast, 1, :group, "Control", "Treatment")
end

b_string_modelrow = @benchmark contrast_modelrow!(
    $buf_string, $(engine_string.contrast), 1, :group, "Control", "Treatment"
) samples=200

println("\nString - contrast_modelrow!:")
println("  Allocations: ", b_string_modelrow.allocs)
println("  Memory:      ", b_string_modelrow.memory, " bytes")

println("\n2. Test contrast_gradient! directly")
println("-"^80)

# Boolean
grad_buf_bool = Vector{Float64}(undef, length(coef(model_bool)))
for _ in 1:10
    contrast_gradient!(grad_buf_bool, engine_bool.contrast, 1, :treated, false, true, engine_bool.β, engine_bool.link)
end

b_bool_gradient = @benchmark contrast_gradient!(
    $grad_buf_bool, $(engine_bool.contrast), 1, :treated, false, true, $(engine_bool.β), $(engine_bool.link)
) samples=200

println("\nBoolean - contrast_gradient!:")
println("  Allocations: ", b_bool_gradient.allocs)
println("  Memory:      ", b_bool_gradient.memory, " bytes")

# String
grad_buf_string = Vector{Float64}(undef, length(coef(model_string)))
for _ in 1:10
    contrast_gradient!(grad_buf_string, engine_string.contrast, 1, :group, "Control", "Treatment", engine_string.β, engine_string.link)
end

b_string_gradient = @benchmark contrast_gradient!(
    $grad_buf_string, $(engine_string.contrast), 1, :group, "Control", "Treatment", $(engine_string.β), $(engine_string.link)
) samples=200

println("\nString - contrast_gradient!:")
println("  Allocations: ", b_string_gradient.allocs)
println("  Memory:      ", b_string_gradient.memory, " bytes")

println("\n3. Analysis")
println("-"^80)

total_bool = b_bool_modelrow.allocs + b_bool_gradient.allocs
total_string = b_string_modelrow.allocs + b_string_gradient.allocs

println("\nPer-row primitive allocations:")
println("  Boolean:")
println("    contrast_modelrow!:  ", b_bool_modelrow.allocs)
println("    contrast_gradient!:  ", b_bool_gradient.allocs)
println("    Total per row:       ", total_bool)
println("\n  String:")
println("    contrast_modelrow!:  ", b_string_modelrow.allocs)
println("    contrast_gradient!:  ", b_string_gradient.allocs)
println("    Total per row:       ", total_string)

println("\nDifference per row: ", total_bool - total_string)

# Calculate for 50 rows (the batch size)
rows_in_batch = 50
total_bool_batch = total_bool * rows_in_batch
total_string_batch = total_string * rows_in_batch

println("\nFor 50-row batch:")
println("  Boolean total:  ", total_bool_batch, " (", total_bool, " per row × 50)")
println("  String total:   ", total_string_batch, " (", total_string, " per row × 50)")
println("  Expected diff:  ", total_bool_batch - total_string_batch)
println("  Observed diff:  700 (from batch kernel test)")

if total_bool_batch - total_string_batch == 700
    println("\nMATCH! The 700 allocations are exactly ", total_bool, " per row × 50 rows")
end

println("\n" * "="^80)
println("Root Cause Identification:")
println("="^80)

if b_bool_modelrow.allocs > b_string_modelrow.allocs
    println("\n contrast_modelrow! allocates ", b_bool_modelrow.allocs, " per call for Bool")
    println("   File: FormulaCompiler/src/evaluation/derivatives/contrasts.jl")
    println("   Function: contrast_modelrow!")
    println("   Issue: Type instability or boxing when handling Bool levels")
end

if b_bool_gradient.allocs > b_string_gradient.allocs
    println("\n contrast_gradient! allocates ", b_bool_gradient.allocs, " per call for Bool")
    println("   File: FormulaCompiler/src/evaluation/derivatives/contrasts.jl")
    println("   Function: contrast_gradient!")
    println("   Issue: Type instability or boxing when handling Bool levels")
end

if total_bool == 0 && total_string == 0
    println("\n Both primitives are 0-alloc!")
    println("   The 700 allocations must be elsewhere in the stack")
end

println("\n" * "="^80)
