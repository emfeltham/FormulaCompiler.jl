# phase1_complete_recursion.jl - True recursive single-term evaluation

###############################################################################
# Complete recursive implementation for arbitrary formula complexity
###############################################################################

using StatsModels
using StatsModels: AbstractTerm, Term, ContinuousTerm, CategoricalTerm, ConstantTerm, 
                   InterceptTerm, FunctionTerm, InteractionTerm, MatrixTerm
using CategoricalArrays: refs
using Tables
using LinearAlgebra

###############################################################################
# Core Recursive Evaluation Functions
###############################################################################

"""
    evaluate_term_recursive!(term::AbstractTerm, data::NamedTuple, output::AbstractMatrix)

Fully recursive single-term evaluation that handles arbitrary complexity without
falling back to the full _cols! system. This is the true implementation for Strategy 4.

# Key Features
- Pure recursion: no dependency on InplaceModeler's _cols! system
- Handles nested functions, complex interactions, arbitrary formula depth
- Maintains numerical accuracy identical to full model matrix construction
- Enables true column-by-column computation for Strategy 4
"""
function evaluate_term_recursive!(term::AbstractTerm, data::NamedTuple, output::AbstractMatrix)
    _evaluate_term_recursive_impl!(term, data, output)
    return output
end

"""
    evaluate_single_column_recursive!(term::AbstractTerm, data::NamedTuple, global_col::Int, local_col::Int, output::AbstractVector)

Recursive single-column evaluation for maximum efficiency.
"""
function evaluate_single_column_recursive!(
    term::AbstractTerm, 
    data::NamedTuple, 
    global_col::Int, 
    local_col::Int,
    output::AbstractVector
)
    w = width(term)
    
    if w == 1 && !(term isa InteractionTerm) && !(term isa CategoricalTerm)
        # Single-column term that supports direct evaluation
        _evaluate_single_column_recursive_impl!(term, data, output)
    else
        # Multi-column term OR term that requires full matrix evaluation
        # This includes: InteractionTerm, CategoricalTerm, and any multi-column terms
        temp_matrix = Matrix{Float64}(undef, length(output), w)
        evaluate_term_recursive!(term, data, temp_matrix)
        copy!(output, view(temp_matrix, :, local_col))
    end
    return output
end

###############################################################################
# Recursive Implementation for Each Term Type
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# Base Cases: Simple Terms (No Recursion Needed)
# ─────────────────────────────────────────────────────────────────────────────

function _evaluate_term_recursive_impl!(term::ContinuousTerm, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1 "ContinuousTerm produces exactly 1 column"
    copy!(view(output, :, 1), data[term.sym])
    return output
end

function _evaluate_single_column_recursive_impl!(term::ContinuousTerm, data::NamedTuple, output::AbstractVector)
    copy!(output, data[term.sym])
    return output
end

function _evaluate_term_recursive_impl!(term::ConstantTerm, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1 "ConstantTerm produces exactly 1 column"
    fill!(view(output, :, 1), term.n)
    return output
end

function _evaluate_single_column_recursive_impl!(term::ConstantTerm, data::NamedTuple, output::AbstractVector)
    fill!(output, term.n)
    return output
end

function _evaluate_term_recursive_impl!(term::InterceptTerm{true}, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1 "InterceptTerm produces exactly 1 column"
    fill!(view(output, :, 1), 1.0)
    return output
end

function _evaluate_single_column_recursive_impl!(term::InterceptTerm{true}, data::NamedTuple, output::AbstractVector)
    fill!(output, 1.0)
    return output
end

function _evaluate_term_recursive_impl!(term::InterceptTerm{false}, data::NamedTuple, output::AbstractMatrix)
    # No columns produced - output should be empty
    @assert size(output, 2) == 0 "InterceptTerm{false} produces no columns"
    return output
end

function _evaluate_term_recursive_impl!(term::Term, data::NamedTuple, output::AbstractMatrix)
    # Untyped Term treated as ContinuousTerm
    @assert size(output, 2) == 1 "Term produces exactly 1 column"
    copy!(view(output, :, 1), data[term.sym])
    return output
end

function _evaluate_single_column_recursive_impl!(term::Term, data::NamedTuple, output::AbstractVector)
    copy!(output, data[term.sym])
    return output
end

# ─────────────────────────────────────────────────────────────────────────────
# Categorical Terms (No Recursion, But Complex Logic)
# ─────────────────────────────────────────────────────────────────────────────

function _evaluate_term_recursive_impl!(term::CategoricalTerm{C,T,N}, data::NamedTuple, output::AbstractMatrix) where {C,T,N}
    @assert size(output, 2) == N "CategoricalTerm should produce $N columns"
    
    v = data[term.sym]
    
    # CRITICAL FIX: Ensure exact compatibility with StatsModels/modelmatrix!
    # 
    # The key insights:
    # 1. Use CategoricalArrays.refs() to get integer codes (1-based)
    # 2. Use the exact contrast matrix from the fitted term
    # 3. Apply contrasts exactly as StatsModels does: output[row, col] = M[level_code, col]
    
    # Get integer level codes (1-based indices into the level pool)
    codes = refs(v)
    
    # Get the contrast matrix that was computed when the model was fitted
    # This matrix M has dimensions (n_levels × n_dummy_columns)
    # where n_levels = length(levels(categorical_var))
    # and n_dummy_columns = N (the number of dummy variables created)
    M = term.contrasts.matrix
    
    # Validate dimensions for debugging
    n_levels, n_dummy_cols = size(M)
    rows = length(codes)
    
    @assert n_dummy_cols == N "Contrast matrix produces $n_dummy_cols columns, expected $N"
    @assert rows == size(output, 1) "Data has $rows observations, output has $(size(output, 1)) rows"
    
    # Apply contrast coding: for each observation, look up its level and apply the contrast
    # This is the EXACT same operation that StatsModels does internally
    @inbounds for r in 1:rows
        level_code = codes[r]  # Which level does observation r have? (1-based)
        
        # Safety check (should never trigger with valid categorical data)
        if level_code < 1 || level_code > n_levels
            error("Invalid level code $level_code for categorical $(term.sym) (valid range: 1-$n_levels)")
        end
        
        # Apply the contrast: copy row level_code from the contrast matrix
        # This gives us the dummy variable values for this level
        @simd for k in 1:N
            output[r, k] = M[level_code, k]
        end
    end
    
    return output
end

# ─────────────────────────────────────────────────────────────────────────────
# Recursive Cases: Complex Terms
# ─────────────────────────────────────────────────────────────────────────────

"""
    _evaluate_term_recursive_impl!(term::FunctionTerm, data::NamedTuple, output::AbstractMatrix)

Recursive evaluation of function terms. Handles arbitrary nesting like:
- log(x)
- x + z  
- log(x + z^2)
- sin(cos(x) + exp(z))
"""
function _evaluate_term_recursive_impl!(term::FunctionTerm, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1 "FunctionTerm produces exactly 1 column"
    
    n = size(output, 1)
    nargs = length(term.args)
    col = view(output, :, 1)
    
    # Recursively evaluate each argument
    arg_buffers = Vector{Vector{Float64}}(undef, nargs)
    
    for (i, arg) in enumerate(term.args)
        arg_buffers[i] = Vector{Float64}(undef, n)
        
        if arg isa ContinuousTerm
            # Optimization: direct access for continuous terms
            copy!(arg_buffers[i], data[arg.sym])
        elseif arg isa ConstantTerm
            # Optimization: direct fill for constants
            fill!(arg_buffers[i], arg.n)
        else
            # General case: recursive evaluation
            _evaluate_single_column_recursive_impl!(arg, data, arg_buffers[i])
        end
    end
    
    # Apply function to evaluated arguments
    @inbounds @simd for i in 1:n
        if nargs == 1
            col[i] = term.f(arg_buffers[1][i])
        elseif nargs == 2
            col[i] = term.f(arg_buffers[1][i], arg_buffers[2][i])
        elseif nargs == 3
            col[i] = term.f(arg_buffers[1][i], arg_buffers[2][i], arg_buffers[3][i])
        else
            # General case: use ntuple for efficiency
            args_tuple = ntuple(j -> arg_buffers[j][i], nargs)
            col[i] = term.f(args_tuple...)
        end
    end
    
    return output
end

function _evaluate_single_column_recursive_impl!(term::FunctionTerm, data::NamedTuple, output::AbstractVector)
    # For function terms, we can evaluate directly into the output vector
    temp_matrix = reshape(output, :, 1)  # View as matrix
    _evaluate_term_recursive_impl!(term, data, temp_matrix)
    return output
end

# Note: InteractionTerm and CategoricalTerm don't need _evaluate_single_column_recursive_impl!
# methods because they always use the full matrix approach in evaluate_single_column_recursive!

"""
    _evaluate_term_recursive_impl!(term::InteractionTerm, data::NamedTuple, output::AbstractMatrix)

Recursive evaluation of interaction terms. Computes the Kronecker product of component terms.
FIXED VERSION: Corrected indexing for categorical interactions.
"""
function _evaluate_term_recursive_impl!(term::InteractionTerm, data::NamedTuple, output::AbstractMatrix)
    n = size(output, 1)
    components = term.terms
    ncomponents = length(components)
    
    # Step 1: Recursively evaluate each component term
    component_matrices = Vector{Matrix{Float64}}(undef, ncomponents)
    component_widths = Vector{Int}(undef, ncomponents)
    
    for (i, component) in enumerate(components)
        comp_width = width(component)
        component_widths[i] = comp_width
        component_matrices[i] = Matrix{Float64}(undef, n, comp_width)
        
        # Recursive call - this handles arbitrary nesting
        evaluate_term_recursive!(component, data, component_matrices[i])
    end
    
    # Step 2: Compute Kronecker product manually with correct indexing
    total_width = prod(component_widths)
    @assert size(output, 2) == total_width "InteractionTerm width mismatch"
    
    # FIXED: Use correct Kronecker product indexing
    compute_kronecker_product_fixed!(component_matrices, component_widths, output)
    
    return output
end

"""
    compute_kronecker_product_fixed!(components, widths, output)

Fixed Kronecker product computation with correct indexing for categorical interactions.
"""
function compute_kronecker_product_fixed!(components::Vector{Matrix{Float64}}, widths::Vector{Int}, output::AbstractMatrix)
    n, total_cols = size(output)
    ncomponents = length(components)
    
    # For each output column, determine which component columns to multiply
    col_idx = 1
    
    # Generate all combinations of component column indices
    ranges = [1:w for w in widths]
    for indices in Iterators.product(ranges...)
        # indices is a tuple like (1, 1), (1, 2), (2, 1), (2, 2) for 2x2 case
        
        @inbounds for row in 1:n
            product = 1.0
            for (comp_idx, col_in_comp) in enumerate(indices)
                product *= components[comp_idx][row, col_in_comp]
            end
            output[row, col_idx] = product
        end
        
        col_idx += 1
    end
    
    return output
end

"""
    compute_kronecker_product!(components::Vector{Matrix{Float64}}, widths::Vector{Int}, output::AbstractMatrix)

Efficiently compute the row-wise Kronecker product of multiple matrices.
This is the core operation for interaction terms.

FIXED VERSION: Improved numerical stability and indexing.
"""
function compute_kronecker_product!(components::Vector{Matrix{Float64}}, widths::Vector{Int}, output::AbstractMatrix)
    n, total_cols = size(output)
    ncomponents = length(components)
    
    # Verify input consistency
    @assert length(widths) == ncomponents
    @assert prod(widths) == total_cols
    
    # Compute stride factors for indexing (these determine column mapping)
    strides = Vector{Int}(undef, ncomponents)
    strides[1] = 1
    for k in 2:ncomponents
        strides[k] = strides[k-1] * widths[k-1]
    end
    
    # Compute Kronecker product row by row
    @inbounds for row in 1:n
        for col in 1:total_cols
            # Decompose global column index into component indices
            remaining = col - 1
            product = 1.0
            
            # Process components in order
            for k in 1:ncomponents
                comp_col = (remaining ÷ strides[k]) % widths[k] + 1
                component_val = components[k][row, comp_col]
                
                # Check for NaN/Inf to help debug numerical issues
                if !isfinite(component_val)
                    product = component_val
                    break
                else
                    product *= component_val
                end
                
                remaining = remaining % strides[k]
            end
            
            output[row, col] = product
        end
    end
    
    return output
end

"""
    _evaluate_term_recursive_impl!(term::MatrixTerm, data::NamedTuple, output::AbstractMatrix)

Recursive evaluation of matrix terms (tuples of terms).
Handles cases like: (x + z) which becomes MatrixTerm([ContinuousTerm(:x), ContinuousTerm(:z)])
"""
function _evaluate_term_recursive_impl!(term::MatrixTerm, data::NamedTuple, output::AbstractMatrix)
    col_offset = 0
    
    for subterm in term.terms
        subterm_width = width(subterm)
        if subterm_width > 0
            subterm_output = view(output, :, col_offset+1:col_offset+subterm_width)
            
            # Recursive call - handles arbitrary nesting
            evaluate_term_recursive!(subterm, data, subterm_output)
            
            col_offset += subterm_width
        end
    end
    
    return output
end

###############################################################################
# Integration with Phase 1 Interface
###############################################################################

"""
    evaluate_term_complete!(term::AbstractTerm, data::NamedTuple, output::AbstractMatrix)

Complete single-term evaluation that handles arbitrary complexity.
This is the improved version that replaces the fallback-based implementation.
"""
function evaluate_term_complete!(term::AbstractTerm, data::NamedTuple, output::AbstractMatrix)
    return evaluate_term_recursive!(term, data, output)
end

"""
    evaluate_single_column_complete!(term::AbstractTerm, data::NamedTuple, global_col::Int, local_col::Int, output::AbstractVector)

Complete single-column evaluation with full recursion support.
"""
function evaluate_single_column_complete!(
    term::AbstractTerm, 
    data::NamedTuple, 
    global_col::Int, 
    local_col::Int,
    output::AbstractVector
)
    return evaluate_single_column_recursive!(term, data, global_col, local_col, output)
end

###############################################################################
# Validation and Testing
###############################################################################

"""
    validate_recursive_evaluation(model::StatisticalModel, data::NamedTuple) -> Bool

Validate that recursive single-term evaluation produces identical results to full model matrix construction.
This is the critical test that proves Strategy 4 will work correctly.
"""
function validate_recursive_evaluation(model::StatisticalModel, data::NamedTuple)
    # Get full model matrix using existing system
    n = length(first(data))
    p = width(fixed_effects_form(model).rhs)
    ipm = InplaceModeler(model, n)
    X_full = Matrix{Float64}(undef, n, p)
    modelmatrix!(ipm, data, X_full)
    
    # Get column mapping
    mapping = enhanced_column_mapping(model)
    
    # Validate each column using recursive single-term evaluation
    all_match = true
    max_overall_diff = 0.0
    
    for col in 1:p
        term, local_col = get_term_for_column(mapping, col)
        
        # Evaluate single column using complete recursive system
        col_result = Vector{Float64}(undef, n)
        try
            evaluate_single_column_complete!(term, data, col, local_col, col_result)
            
            # Compare with full matrix column
            full_col = X_full[:, col]
            max_diff = maximum(abs.(col_result .- full_col))
            max_overall_diff = max(max_overall_diff, max_diff)
            
            if max_diff > 1e-12
                @warn "Column $col mismatch: max difference = $max_diff" term
                all_match = false
            end
        catch e
            @error "Failed to evaluate column $col recursively" term exception=e
            all_match = false
        end
    end
    
    if all_match
        println("✅ Recursive evaluation passed: max difference = $(max_overall_diff)")
    else
        println("❌ Recursive evaluation failed")
    end
    
    return all_match
end

"""
    test_complex_recursive_formulas()

Test recursive evaluation on increasingly complex formulas.
"""
function test_complex_recursive_formulas()
    println("Testing recursive evaluation on complex formulas...")
    
    n = 100
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        w = abs.(randn(n)) .+ 0.1,  # Ensure positive for log
        group = rand(["A", "B", "C"], n),
        y = randn(n)
    )
    
    # Increasingly complex formulas
    complex_formulas = [
        @formula(y ~ x),                                    # Simple
        @formula(y ~ x + z),                               # Multiple terms
        @formula(y ~ log(w)),                              # Single function
        @formula(y ~ x + log(w)),                          # Mixed
        @formula(y ~ log(w) + sqrt(w)),                    # Multiple functions
        @formula(y ~ x & z),                               # Simple interaction
        @formula(y ~ x & log(w)),                          # Function in interaction
        @formula(y ~ log(w) & sqrt(w)),                    # Function-function interaction
        @formula(y ~ x + z + x & z),                       # Main effects + interaction
        @formula(y ~ x + log(w) + x & log(w)),             # Complex mixed
        @formula(y ~ x & z & group),                       # Three-way interaction
        @formula(y ~ log(x + w) + x & z),                  # Nested function + interaction
        @formula(y ~ (x + z) & (log(w) + sqrt(w))),        # Complex nested
    ]
    
    println("Testing $(length(complex_formulas)) formulas of increasing complexity:")
    
    for (i, formula) in enumerate(complex_formulas)
        print("  Formula $i: ")
        try
            model = lm(formula, df)
            data = Tables.columntable(df)
            
            is_valid = validate_recursive_evaluation(model, data)
            
            if is_valid
                println("✅ PASSED - $formula")
            else
                println("❌ FAILED - $formula")
                return false
            end
        catch e
            println("❌ ERROR - $formula: $e")
            return false
        end
    end
    
    println("\n🎉 All complex formulas handled correctly by recursive evaluation!")
    println("✅ No fallbacks needed - true single-term evaluation achieved")
    return true
end

###############################################################################
# Missing Column Mapping Functions (from basic implementation)
###############################################################################

"""
    get_term_for_column(mapping::ColumnMapping, col::Int) -> (AbstractTerm, Int)

Find which term generates a specific global column and the local column index within that term.
"""
function get_term_for_column(mapping::ColumnMapping, col::Int)
    for (term, range) in mapping.term_info
        if col in range
            local_col = col - first(range) + 1
            return (term, local_col)
        end
    end
    error("Column $col not found in mapping")
end

"""
    get_terms_for_columns(mapping::ColumnMapping, cols::Vector{Int}) -> Dict{AbstractTerm, Vector{Tuple{Int,Int}}}

Group columns by the terms that generate them, returning local column indices.
"""
function get_terms_for_columns(mapping::ColumnMapping, cols::Vector{Int})
    terms_map = Dict{AbstractTerm, Vector{Tuple{Int,Int}}}()
    
    for col in cols
        term, local_col = get_term_for_column(mapping, col)
        
        if !haskey(terms_map, term)
            terms_map[term] = Tuple{Int,Int}[]
        end
        push!(terms_map[term], (col, local_col))
    end
    
    return terms_map
end

"""
    get_variable_ranges(mapping::ColumnMapping, sym::Symbol) -> Vector{UnitRange{Int}}

Get ALL column ranges where a variable appears.
"""
function get_variable_ranges(mapping::ColumnMapping, sym::Symbol)
    return get(mapping.symbol_to_ranges, sym, UnitRange{Int}[])
end

"""
    get_terms_involving_variable(mapping::ColumnMapping, sym::Symbol) -> Vector{Tuple{AbstractTerm, UnitRange{Int}}}

Get all terms that involve a specific variable, along with their column ranges.
"""
function get_terms_involving_variable(mapping::ColumnMapping, sym::Symbol)
    result = Tuple{AbstractTerm, UnitRange{Int}}[]
    ranges = get_variable_ranges(mapping, sym)
    
    for range in ranges
        if haskey(mapping.range_to_terms, range)
            for term in mapping.range_to_terms[range]
                push!(result, (term, range))
            end
        end
    end
    
    return result
end

# Note: build_column_mapping() should come from the existing termmapping.jl in EfficientModelMatrices
# We'll assume it's available from the existing system

###############################################################################
# Aliases for Compatibility with Testing Files
###############################################################################

"""
    evaluate_term!(term::AbstractTerm, data::NamedTuple, output::AbstractMatrix, imp::Union{InplaceModeler,Nothing})

Interface-compatible version that matches testing file expectations.
The ipm parameter is ignored since recursive evaluation doesn't need it.
"""
function evaluate_term!(term::AbstractTerm, data::NamedTuple, output::AbstractMatrix, ipm::Union{InplaceModeler,Nothing})
    return evaluate_term_recursive!(term, data, output)
end

"""
    evaluate_single_column!(term::AbstractTerm, data::NamedTuple, global_col::Int, local_col::Int, output::AbstractVector, ipm::Union{InplaceModeler,Nothing})

Interface-compatible version that matches testing file expectations.
The ipm parameter is ignored since recursive evaluation doesn't need it.
"""
function evaluate_single_column!(
    term::AbstractTerm, 
    data::NamedTuple, 
    global_col::Int, 
    local_col::Int,
    output::AbstractVector,
    ipm::Union{InplaceModeler,Nothing}
)
    return evaluate_single_column_recursive!(term, data, global_col, local_col, output)
end

# Note: evaluate_term_complete! and evaluate_single_column_complete! are defined above
# The evaluate_term! and evaluate_single_column! functions ARE the complete versions

"""
    enhanced_column_mapping(model::StatisticalModel) -> ColumnMapping

Create ColumnMapping with validation against existing model matrix.
"""
function enhanced_column_mapping(model::StatisticalModel)
    X_fitted = modelmatrix(model)
    rhs = fixed_effects_form(model).rhs
    mapping = build_column_mapping(rhs)
    
    # Validate mapping matches fitted model
    if mapping.total_columns != size(X_fitted, 2)
        error("Column mapping mismatch: mapping has $(mapping.total_columns) columns, model matrix has $(size(X_fitted, 2)) columns")
    end
    
    return mapping
end

"""
    get_all_variable_columns(mapping::ColumnMapping, sym::Symbol) -> Vector{Int}

Get all column indices where a variable appears.
"""
function get_all_variable_columns(mapping::ColumnMapping, sym::Symbol)
    ranges = get_variable_ranges(mapping, sym)
    cols = Int[]
    for range in ranges
        append!(cols, collect(range))
    end
    return unique!(sort!(cols))
end

"""
    validate_single_term_evaluation(model::StatisticalModel, data::NamedTuple) -> Bool

Validate that single-term evaluation produces the same results as full model matrix construction.
This function is expected by the testing files.
"""
function validate_single_term_evaluation(model::StatisticalModel, data::NamedTuple)
    return validate_recursive_evaluation(model, data)
end

# Export all functions needed by testing files
export evaluate_term!, evaluate_single_column!
export enhanced_column_mapping, get_all_variable_columns
export get_term_for_column, get_terms_for_columns
export get_variable_ranges, get_terms_involving_variable
export validate_recursive_evaluation, validate_single_term_evaluation
export test_complex_recursive_formulas, compute_kronecker_product!
