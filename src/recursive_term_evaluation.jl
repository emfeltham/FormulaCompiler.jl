# recursive_term_evaluation.jl - Add to EfficientModelMatrices.jl/termmapping.jl

###############################################################################
# Phase 1 Integration: Add these functions to EfficientModelMatrices.jl
# Append this content to the existing termmapping.jl file
###############################################################################

# Phase 1 Recursive Evaluation Functions (from phase1_complete_recursion.jl)
# Add all the recursive evaluation functions here...

"""
    enhanced_column_mapping(model::StatisticalModel) -> ColumnMapping

Create ColumnMapping with validation against existing model matrix.
Required for Strategy 4 integration.
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

Get a flat vector of ALL column indices where a variable appears.
Required for Strategy 4 affected columns analysis.
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
    evaluate_single_column!(term::AbstractTerm, data::NamedTuple, global_col::Int, local_col::Int, output::AbstractVector, imp::Union{InplaceModeler,Nothing})

Phase 1 recursive single-column evaluation.
Core function for Strategy 4 selective computation.
"""
function evaluate_single_column!(
    term::AbstractTerm, 
    data::NamedTuple, 
    global_col::Int, 
    local_col::Int,
    output::AbstractVector,
    imp::Union{InplaceModeler,Nothing}
)
    return evaluate_single_column_recursive!(term, data, global_col, local_col, output)
end

# Phase 1 Recursive Evaluation Implementation
# (Copy all functions from phase1_complete_recursion.jl here)

function evaluate_single_column_recursive!(
    term::AbstractTerm, 
    data::NamedTuple, 
    global_col::Int, 
    local_col::Int,
    output::AbstractVector
)
    w = width(term)
    
    if w == 1 && !(term isa InteractionTerm) && !(term isa CategoricalTerm)
        _evaluate_single_column_recursive_impl!(term, data, output)
    else
        temp_matrix = Matrix{Float64}(undef, length(output), w)
        evaluate_term_recursive!(term, data, temp_matrix)
        copy!(output, view(temp_matrix, :, local_col))
    end
    return output
end

function evaluate_term_recursive!(term::AbstractTerm, data::NamedTuple, output::AbstractMatrix)
    _evaluate_term_recursive_impl!(term, data, output)
    return output
end

# Base cases: Simple terms
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
    @assert size(output, 2) == 0 "InterceptTerm{false} produces no columns"
    return output
end

function _evaluate_term_recursive_impl!(term::Term, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1 "Term produces exactly 1 column"
    copy!(view(output, :, 1), data[term.sym])
    return output
end

function _evaluate_single_column_recursive_impl!(term::Term, data::NamedTuple, output::AbstractVector)
    copy!(output, data[term.sym])
    return output
end

# Categorical terms
function _evaluate_term_recursive_impl!(term::CategoricalTerm{C,T,N}, data::NamedTuple, output::AbstractMatrix) where {C,T,N}
    @assert size(output, 2) == N "CategoricalTerm should produce $N columns"
    
    v = data[term.sym]
    codes = refs(v)
    M = term.contrasts.matrix
    n_levels, n_dummy_cols = size(M)
    rows = length(codes)
    
    @assert n_dummy_cols == N "Contrast matrix produces $n_dummy_cols columns, expected $N"
    @assert rows == size(output, 1) "Data has $rows observations, output has $(size(output, 1)) rows"
    
    @inbounds for r in 1:rows
        level_code = codes[r]
        if level_code < 1 || level_code > n_levels
            error("Invalid level code $level_code for categorical $(term.sym) (valid range: 1-$n_levels)")
        end
        @simd for k in 1:N
            output[r, k] = M[level_code, k]
        end
    end
    
    return output
end

# Function terms
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
            copy!(arg_buffers[i], data[arg.sym])
        elseif arg isa ConstantTerm
            fill!(arg_buffers[i], arg.n)
        else
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
            args_tuple = ntuple(j -> arg_buffers[j][i], nargs)
            col[i] = term.f(args_tuple...)
        end
    end
    
    return output
end

function _evaluate_single_column_recursive_impl!(term::FunctionTerm, data::NamedTuple, output::AbstractVector)
    temp_matrix = reshape(output, :, 1)
    _evaluate_term_recursive_impl!(term, data, temp_matrix)
    return output
end

# Interaction terms
function _evaluate_term_recursive_impl!(term::InteractionTerm, data::NamedTuple, output::AbstractMatrix)
    n = size(output, 1)
    components = term.terms
    ncomponents = length(components)
    
    # Recursively evaluate each component term
    component_matrices = Vector{Matrix{Float64}}(undef, ncomponents)
    component_widths = Vector{Int}(undef, ncomponents)
    
    for (i, component) in enumerate(components)
        comp_width = width(component)
        component_widths[i] = comp_width
        component_matrices[i] = Matrix{Float64}(undef, n, comp_width)
        evaluate_term_recursive!(component, data, component_matrices[i])
    end
    
    # Compute Kronecker product
    total_width = prod(component_widths)
    @assert size(output, 2) == total_width "InteractionTerm width mismatch"
    
    compute_kronecker_product_fixed!(component_matrices, component_widths, output)
    
    return output
end

function compute_kronecker_product_fixed!(components::Vector{Matrix{Float64}}, widths::Vector{Int}, output::AbstractMatrix)
    n, total_cols = size(output)
    ncomponents = length(components)
    
    col_idx = 1
    ranges = [1:w for w in widths]
    for indices in Iterators.product(ranges...)
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

# Matrix terms
function _evaluate_term_recursive_impl!(term::MatrixTerm, data::NamedTuple, output::AbstractMatrix)
    col_offset = 0
    
    for subterm in term.terms
        subterm_width = width(subterm)
        if subterm_width > 0
            subterm_output = view(output, :, col_offset+1:col_offset+subterm_width)
            evaluate_term_recursive!(subterm, data, subterm_output)
            col_offset += subterm_width
        end
    end
    
    return output
end

# Export Phase 1 functions
# export enhanced_column_mapping, get_all_variable_columns, evaluate_single_column!
# export evaluate_single_column_recursive!, evaluate_term_recursive!