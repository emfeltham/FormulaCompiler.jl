# docs_examples_extracted.jl
# Run with: julia --project=. test/docs_examples_extracted.jl > test/docs_examples_extracted.txt 2>&1

using FormulaCompiler, GLM, DataFrames, CategoricalArrays, Tables

println("=" ^ 80)
println("Extracted Documentation Examples - FormulaCompiler.jl")
println("=" ^ 80)
println()

# =============================================================================
# GLOBAL SETUP - Shared test fixtures for all examples
# =============================================================================
println("Setting up global test fixtures...")
println()

# Create test data (used by all examples)
df = DataFrame(
    y = randn(1000),
    x = randn(1000),
    z = abs.(randn(1000)) .+ 0.1,
    group = categorical(rand(["A", "B", "C"], 1000))
)

# Fit model
model = lm(@formula(y ~ x * group + log(z)), df)

# Compile formula once (the "compile once, use many times" pattern)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Pre-allocate common output vectors
row_vec = Vector{Float64}(undef, length(compiled))
output = Vector{Float64}(undef, length(compiled))

println("Global fixtures ready: df, model, data, compiled, row_vec, output")
println()

# =============================================================================
# README.md
# =============================================================================
println("\n", "=" ^ 80)
println("Source: README.md")
println("=" ^ 80, "\n")

# Lines 33-53: Basic Usage
println("# Basic Usage (lines 33-53)")
try
    # Evaluate rows without allocation
    compiled(row_vec, data, 1)  # Zero allocations after warmup

    println("✓ Basic usage example passed")
catch e
    println("✗ Basic usage example failed: ", e)
end

# Lines 61-76: Direct Compilation Interface
println("\n# Direct Compilation Interface (lines 61-76)")
try
    # Evaluate individual rows
    compiled(row_vec, data, 1)    # Row 1
    compiled(row_vec, data, 100)  # Row 100

    # Evaluate multiple rows
    matrix = Matrix{Float64}(undef, 10, length(compiled))
    for i in 1:10
        compiled(view(matrix, i, :), data, i)
    end

    println("✓ Direct compilation interface example passed")
catch e
    println("✗ Direct compilation interface example failed: ", e)
end

# Lines 82-92: Convenience Interface
println("\n# Convenience Interface (lines 82-92)")
try
    # Single row (allocating)
    row_values = modelrow(model, data, 1)

    # Multiple rows (allocating)
    matrix = modelrow(model, data, [1, 5, 10, 50])

    # In-place with automatic caching
    row_vec_local = Vector{Float64}(undef, size(modelmatrix(model), 2))
    modelrow!(row_vec_local, model, data, 1)

    println("✓ Convenience interface example passed")
catch e
    println("✗ Convenience interface example failed: ", e)
end

# Lines 98-102: Object-Based Interface
println("\n# Object-Based Interface (lines 98-102)")
try
    evaluator = ModelRowEvaluator(model, df)
    result = evaluator(1)           # Row 1
    evaluator(row_vec, 1)          # In-place evaluation

    println("✓ Object-based interface example passed")
catch e
    println("✗ Object-based interface example failed: ", e)
end

# Lines 332-342: Batch Operations
println("\n# Batch Operations (lines 332-342)")
try
    # Efficient: batch evaluation with pre-allocation
    matrix = Matrix{Float64}(undef, 10, length(compiled))
    for i in 1:10
        compiled(view(matrix, i, :), data, i)
    end

    # Inefficient: repeated allocation
    results = [modelrow(model, data, i) for i in 1:10]

    println("✓ Batch operations example passed")
catch e
    println("✗ Batch operations example failed: ", e)
end

# =============================================================================
# docs/src/getting_started.md
# =============================================================================
println("\n", "=" ^ 80)
println("Source: docs/src/getting_started.md")
println("=" ^ 80, "\n")

# Lines 123-131: Batch Evaluation
println("\n# Batch Evaluation (lines 123-131)")
try
    # Pre-allocate matrix
    matrix = Matrix{Float64}(undef, 10, length(compiled))

    # Evaluate rows 1-10 in batch
    for i in 1:10
        compiled(view(matrix, i, :), data, i)
    end

    println("✓ Batch evaluation example passed")
catch e
    println("✗ Batch evaluation example failed: ", e)
end

# =============================================================================
# docs/src/guide/basic_usage.md
# =============================================================================
println("\n", "=" ^ 80)
println("Source: docs/src/guide/basic_usage.md")
println("=" ^ 80, "\n")

# Lines 79-95: Multiple Row Evaluation
println("\n# Multiple Row Evaluation (lines 79-95)")
try
    # Pre-allocate matrix for multiple rows
    n_rows = 100
    matrix = Matrix{Float64}(undef, n_rows, length(compiled))

    # Evaluate multiple rows efficiently
    for i in 1:n_rows
        compiled(view(matrix, i, :), data, i)
    end

    # Or specific rows
    specific_rows = [1, 5, 10, 50, 100]
    matrix_subset = Matrix{Float64}(undef, length(specific_rows), length(compiled))
    for (idx, row) in enumerate(specific_rows)
        compiled(view(matrix_subset, idx, :), data, row)
    end

    println("✓ Multiple row evaluation example passed")
catch e
    println("✗ Multiple row evaluation example failed: ", e)
end

# Lines 271-287: Large Dataset Considerations
println("\n# Large Dataset Considerations (lines 271-287)")
try
    # For very large datasets, process in chunks
    chunk_size = 100  # Reduced for test
    n_chunks = 2  # Reduced for test

    results = Matrix{Float64}(undef, chunk_size * n_chunks, length(compiled))

    for chunk in 1:n_chunks
        start_idx = (chunk - 1) * chunk_size + 1
        end_idx = min(chunk * chunk_size, 1000)

        # Evaluate each row in the chunk
        for i in start_idx:end_idx
            compiled(view(results, i, :), data, i)
        end
    end

    println("✓ Large dataset considerations example passed")
catch e
    println("✗ Large dataset considerations example failed: ", e)
end

# =============================================================================
# docs/src/guide/performance.md
# =============================================================================
println("\n", "=" ^ 80)
println("Source: docs/src/guide/performance.md")
println("=" ^ 80, "\n")

# Lines 68-87: Pre-allocation Strategies
println("\n# Pre-allocation Strategies (lines 68-87)")
try
    # For batch processing
    n_rows = 10  # Reduced for test
    batch_matrix = Matrix{Float64}(undef, n_rows, length(compiled))

    # Reuse across operations
    total_rows = 20  # Reduced for test
    for batch_start in 1:n_rows:total_rows
        batch_end = min(batch_start + n_rows - 1, total_rows)

        # Evaluate each row in the batch
        for i in batch_start:batch_end
            idx = i - batch_start + 1
            compiled(view(batch_matrix, idx, :), data, i)
        end
    end

    println("✓ Pre-allocation strategies example passed")
catch e
    println("✗ Pre-allocation strategies example failed: ", e)
end

# Lines 241-268: Chunked Processing
println("\n# Chunked Processing (lines 241-268)")
try
    function process_large_dataset_efficiently(model, data, chunk_size=100)  # Reduced for test
        compiled = compile_formula(model, data)
        n_rows = Tables.rowcount(data)
        n_cols = length(compiled)

        # Pre-allocate chunk matrix
        chunk_matrix = Matrix{Float64}(undef, chunk_size, n_cols)

        results = Vector{Matrix{Float64}}()

        for start_idx in 1:chunk_size:min(200, n_rows)  # Reduced for test
            end_idx = min(start_idx + chunk_size - 1, n_rows)
            actual_chunk_size = end_idx - start_idx + 1

            # Zero-allocation batch evaluation
            for (chunk_row, data_row) in enumerate(start_idx:end_idx)
                compiled(view(chunk_matrix, chunk_row, :), data, data_row)
            end

            # Store results (this allocates, but unavoidable for storage)
            chunk_view = view(chunk_matrix, 1:actual_chunk_size, :)
            push!(results, copy(chunk_view))
        end

        return results
    end

    chunk_results = process_large_dataset_efficiently(model, data, 100)

    println("✓ Chunked processing example passed")
catch e
    println("✗ Chunked processing example failed: ", e)
end

# =============================================================================
# docs/src/guide/advanced_features.md
# =============================================================================
println("\n", "=" ^ 80)
println("Source: docs/src/guide/advanced_features.md")
println("=" ^ 80, "\n")

# Lines 620-645: Batch Processing Large Datasets
println("\n# Batch Processing Large Datasets (lines 620-645)")
try
    function process_large_dataset(model, data, batch_size=100)  # Reduced for test
        compiled = compile_formula(model, data)
        n_rows = Tables.rowcount(data)
        n_cols = length(compiled)

        results = Vector{Matrix{Float64}}()

        for start_idx in 1:batch_size:min(200, n_rows)  # Reduced for test
            end_idx = min(start_idx + batch_size - 1, n_rows)
            batch_size_actual = end_idx - start_idx + 1

            batch_result = Matrix{Float64}(undef, batch_size_actual, n_cols)

            # Evaluate each row in the batch
            for (batch_idx, data_idx) in enumerate(start_idx:end_idx)
                compiled(view(batch_result, batch_idx, :), data, data_idx)
            end

            push!(results, batch_result)
        end

        return results
    end

    large_results = process_large_dataset(model, data, 100)

    println("✓ Batch processing large datasets example passed")
catch e
    println("✗ Batch processing large datasets example failed: ", e)
end

# =============================================================================
# docs/src/examples.md
# =============================================================================
println("\n", "=" ^ 80)
println("Source: docs/src/examples.md")
println("=" ^ 80, "\n")

# Lines 36-55: Counterfactual Analysis
println("\n# Counterfactual Analysis (lines 36-55)")
try
    # Single policy scenario using direct data modification
    n_rows = length(data.x)
    data_policy = merge(data, (x = fill(2.0, n_rows), group = fill("A", n_rows)))
    compiled(output, data_policy, 1)  # Evaluate with modified data

    # Multi-scenario analysis
    x_values = [-1.0, 0.0, 1.0]
    group_values = ["A", "B"]
    results = Matrix{Float64}(undef, 6, length(compiled))  # 3×2 = 6 scenarios

    scenario_idx = 1
    for x_val in x_values
        for group_val in group_values
            data_scenario = merge(data, (x = fill(x_val, n_rows), group = fill(group_val, n_rows)))
            compiled(view(results, scenario_idx, :), data_scenario, 1)
            scenario_idx += 1
        end
    end

    println("✓ Counterfactual analysis example passed")
catch e
    println("✗ Counterfactual analysis example failed: ", e)
end

# Lines 74-81: Batch Processing
println("\n# Batch Processing (lines 74-81)")
try
    # Multiple rows at once
    n_rows = 50
    results = Matrix{Float64}(undef, n_rows, length(compiled))
    for i in 1:n_rows
        compiled(view(results, i, :), data, i)  # Zero allocations
    end

    println("✓ Batch processing example passed")
catch e
    println("✗ Batch processing example failed: ", e)
end

println("\n", "=" ^ 80)
println("Documentation Examples Extraction Complete")
println("Note: All batch modelrow! calls replaced with loop-based syntax")
println("All examples use shared global fixtures (compile once, use many times)")
println("=" ^ 80)
