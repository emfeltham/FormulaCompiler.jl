# termmapping.jl - Complete Column Mapping with Phase 1 Evaluation

###############################################################################
# ColumnMapping Infrastructure
###############################################################################

"""
    ColumnMapping

Complete mapping from variable symbols to model matrix column ranges with term information.
"""
struct ColumnMapping
    symbol_to_ranges::Dict{Symbol, Vector{UnitRange{Int}}}      # Variable → column ranges
    range_to_terms::Dict{UnitRange{Int}, Vector{AbstractTerm}}  # Range → terms that generate it
    term_to_range::Dict{AbstractTerm, UnitRange{Int}}           # Term → its column range
    total_columns::Int                                          # Total model matrix columns
    term_info::Vector{Tuple{AbstractTerm, UnitRange{Int}}}     # Ordered (term, range) pairs
    model::Union{StatisticalModel,Nothing}                     # Reference to fitted model
end

"""
    build_column_mapping(rhs::AbstractTerm, model::Union{StatisticalModel,Nothing}=nothing) -> ColumnMapping

Build complete column mapping from formula RHS with term evaluation capability.
"""
function build_column_mapping(rhs::AbstractTerm, model::Union{StatisticalModel,Nothing}=nothing)
    symbol_to_ranges = Dict{Symbol, Vector{UnitRange{Int}}}()
    range_to_terms = Dict{UnitRange{Int}, Vector{AbstractTerm}}()
    term_to_range = Dict{AbstractTerm, UnitRange{Int}}()
    term_info = Tuple{AbstractTerm, UnitRange{Int}}[]
    
    current_col = Ref(1)
    _map_columns_recursive!(symbol_to_ranges, range_to_terms, term_to_range, term_info, rhs, current_col)
    
    total_cols = current_col[] - 1
    return ColumnMapping(symbol_to_ranges, range_to_terms, term_to_range, total_cols, term_info, model)
end

"""
    _map_columns_recursive!(mappings..., term, col_ref)

Recursively map all terms to their column ranges.
"""
function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info, 
                                term::AbstractTerm, col_ref::Ref{Int})
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        # Record this term and its range
        push!(term_info, (term, range))
        term_to_range[term] = range
        
        # Get all variables this term depends on
        vars = collect_termvars_recursive(term)
        
        # Map each variable to this range
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        # Record which terms generate this range
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

# Specialized recursive methods
function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info,
                                term::MatrixTerm, col_ref::Ref{Int})
    for t in term.terms
        _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info, t, col_ref)
    end
    return col_ref[]
end

function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info,
                                terms::Tuple, col_ref::Ref{Int})
    for t in terms
        _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info, t, col_ref)
    end
    return col_ref[]
end

function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info,
                                term::InteractionTerm, col_ref::Ref{Int})
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        push!(term_info, (term, range))
        term_to_range[term] = range
        
        # Collect variables from ALL sub-terms
        vars = Set{Symbol}()
        for subterm in term.terms
            union!(vars, collect_termvars_recursive(subterm))
        end
        
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info,
                                term::FunctionTerm, col_ref::Ref{Int})
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        push!(term_info, (term, range))
        term_to_range[term] = range
        
        vars = collect_termvars_recursive(term)
        
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

# Handle ZScoredTerm from StandardizedPredictors.jl (if available)

function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info,
                                term::ZScoredTerm, col_ref::Ref{Int})
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        
        push!(term_info, (term, range))
        term_to_range[term] = range
        
        # Collect variables from the wrapped term
        vars = collect_termvars_recursive(term.term)
        
        for var in vars
            if !haskey(sym_to_ranges, var)
                sym_to_ranges[var] = UnitRange{Int}[]
            end
            push!(sym_to_ranges[var], range)
        end
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    
    return col_ref[]
end

# Base cases that don't contribute variables
function _map_columns_recursive!(sym_to_ranges, range_to_terms, term_to_range, term_info,
                                term::Union{InterceptTerm, ConstantTerm}, col_ref::Ref{Int})
    w = width(term)
    if w > 0
        range = col_ref[]:col_ref[]+w-1
        col_ref[] += w
        push!(term_info, (term, range))
        term_to_range[term] = range
        
        if !haskey(range_to_terms, range)
            range_to_terms[range] = AbstractTerm[]
        end
        push!(range_to_terms[range], term)
    end
    return col_ref[]
end

###############################################################################
# Variable Collection Functions
###############################################################################

"""
    collect_termvars_recursive(term::AbstractTerm) -> Set{Symbol}

Recursively collect ALL variable symbols that a term depends on.
"""
function collect_termvars_recursive(term::AbstractTerm)
    vars = Set{Symbol}()
    _collect_vars_recursive!(vars, term)
    return vars
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::Term)
    push!(vars, term.sym)
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::ContinuousTerm)
    push!(vars, term.sym)
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::CategoricalTerm)
    push!(vars, term.sym)
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::FunctionTerm)
    for arg in term.args
        _collect_vars_recursive!(vars, arg)
    end
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::InteractionTerm)
    for subterm in term.terms
        _collect_vars_recursive!(vars, subterm)
    end
end

function _collect_vars_recursive!(vars::Set{Symbol}, term::MatrixTerm)
    for subterm in term.terms
        _collect_vars_recursive!(vars, subterm)
    end
end

function _collect_vars_recursive!(vars::Set{Symbol}, terms::Tuple)
    for term in terms
        _collect_vars_recursive!(vars, term)
    end
end

# Handle ZScoredTerm from StandardizedPredictors.jl (if available)

function _collect_vars_recursive!(vars::Set{Symbol}, term::ZScoredTerm)
    # Delegate to the wrapped term
    _collect_vars_recursive!(vars, term.term)
end

# Base cases
function _collect_vars_recursive!(vars::Set{Symbol}, term::Union{InterceptTerm, ConstantTerm})
    # Do nothing - these don't contribute variables
end

# Fallback
function _collect_vars_recursive!(vars::Set{Symbol}, term::AbstractTerm)
    try
        for var in termvars(term)
            push!(vars, var)
        end
    catch
        @warn "Could not extract variables from term of type $(typeof(term)): $term"
    end
end

###############################################################################
# Lookup Functions
###############################################################################

"""
    get_variable_ranges(mapping::ColumnMapping, sym::Symbol) -> Vector{UnitRange{Int}}

Get ALL column ranges where a variable appears.
"""
function get_variable_ranges(mapping::ColumnMapping, sym::Symbol)
    return get(mapping.symbol_to_ranges, sym, UnitRange{Int}[])
end

"""
    get_all_variable_columns(mapping::ColumnMapping, sym::Symbol) -> Vector{Int}

Get flat vector of ALL column indices where a variable appears.
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
    get_term_for_column(mapping::ColumnMapping, col::Int) -> (AbstractTerm, Int)

Find which term generates a specific column and the local column index within that term.
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

Group columns by terms that generate them, returning (global_col, local_col) pairs.
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

###############################################################################
# Phase 1 Single Column Evaluation
###############################################################################

"""
    evaluate_single_column!(term::AbstractTerm, data::NamedTuple, global_col::Int, local_col::Int, output::AbstractVector, ipm::Union{InplaceModeler,Nothing})

Evaluate a single column of a term with given data.
"""
function evaluate_single_column!(
    term::AbstractTerm, 
    data::NamedTuple, 
    global_col::Int, 
    local_col::Int,
    output::AbstractVector,
    ipm::Union{InplaceModeler,Nothing}=nothing
)
    w = width(term)
    
    if w == 1
        # Single column term - evaluate directly into output
        _evaluate_single_column_direct!(term, data, output)
    elseif w > 1
        # Multi-column term - evaluate all columns, extract the one we need
        temp_matrix = Matrix{Float64}(undef, length(output), w)
        _evaluate_term_full!(term, data, temp_matrix)
        copy!(output, view(temp_matrix, :, local_col))
    else
        error("Term $term has zero width")
    end
    
    return output
end

###############################################################################
# Direct Single Column Evaluation (width=1 terms)
###############################################################################

function _evaluate_single_column_direct!(term::ContinuousTerm, data::NamedTuple, output::AbstractVector)
    copy!(output, data[term.sym])
end

function _evaluate_single_column_direct!(term::Term, data::NamedTuple, output::AbstractVector)
    copy!(output, data[term.sym])
end

function _evaluate_single_column_direct!(term::ConstantTerm, data::NamedTuple, output::AbstractVector)
    fill!(output, term.n)
end

function _evaluate_single_column_direct!(term::InterceptTerm{true}, data::NamedTuple, output::AbstractVector)
    fill!(output, 1.0)
end

function _evaluate_single_column_direct!(term::FunctionTerm, data::NamedTuple, output::AbstractVector)
    @assert width(term) == 1 "FunctionTerm direct evaluation only for width=1"
    
    n = length(output)
    nargs = length(term.args)
    
    # Evaluate each argument
    arg_buffers = [Vector{Float64}(undef, n) for _ in 1:nargs]
    
    for (i, arg) in enumerate(term.args)
        _evaluate_single_column_direct!(arg, data, arg_buffers[i])
    end
    
    # Apply function
    @inbounds for i in 1:n
        if nargs == 1
            output[i] = term.f(arg_buffers[1][i])
        elseif nargs == 2
            output[i] = term.f(arg_buffers[1][i], arg_buffers[2][i])
        elseif nargs == 3
            output[i] = term.f(arg_buffers[1][i], arg_buffers[2][i], arg_buffers[3][i])
        else
            args_tuple = ntuple(j -> arg_buffers[j][i], nargs)
            output[i] = term.f(args_tuple...)
        end
    end
    
    return output
end

function _evaluate_single_column_direct!(term::InteractionTerm, data::NamedTuple, output::AbstractVector)
    # InteractionTerm should only use single-column evaluation if it actually produces 1 column
    term_width = width(term)
    if term_width != 1
        error("InteractionTerm has width $term_width, cannot evaluate as single column. Use _evaluate_term_full! instead.")
    end
    
    # For single-column interaction, compute the product
    components = term.terms
    
    # Initialize output to 1.0
    fill!(output, 1.0)
    
    # Multiply by each component
    temp_col = Vector{Float64}(undef, length(output))
    for component in components
        _evaluate_single_column_direct!(component, data, temp_col)
        @inbounds @simd for i in 1:length(output)
            output[i] *= temp_col[i]
        end
    end
    
    return output
end

# Add fallback for InterceptTerm{false}
function _evaluate_single_column_direct!(term::InterceptTerm{false}, data::NamedTuple, output::AbstractVector)
    error("InterceptTerm{false} should not be evaluated as it produces no columns")
end

# Add fallback for any AbstractTerm not specifically handled
function _evaluate_single_column_direct!(term::AbstractTerm, data::NamedTuple, output::AbstractVector)
    # Fallback: use full term evaluation for single column
    if width(term) == 1
        temp_matrix = reshape(output, :, 1)
        _evaluate_term_full!(term, data, temp_matrix)
        return output
    else
        error("Cannot evaluate $(typeof(term)) with width $(width(term)) as single column")
    end
end


function _evaluate_single_column_direct!(term::ZScoredTerm, data::NamedTuple, output::AbstractVector)
    @assert width(term) == 1 "ZScoredTerm direct evaluation only for width=1"
    
    # Evaluate underlying term first
    _evaluate_single_column_direct!(term.term, data, output)
    
    # Apply Z-score transformation in-place
    _apply_zscore_single_column!(output, term.center, term.scale)
    
    return output
end

###############################################################################
# Full Term Evaluation (multi-column terms)
###############################################################################

function _evaluate_term_full!(term::CategoricalTerm{C,T,N}, data::NamedTuple, output::AbstractMatrix) where {C,T,N}
    @assert size(output, 2) == N "CategoricalTerm should produce $N columns"
    
    v = data[term.sym]
    M = term.contrasts.matrix
    rows = length(v)
    
    @assert rows == size(output, 1) "Data has $rows observations, output has $(size(output, 1)) rows"
    
    # Handle both CategoricalArray and plain vectors
    if isa(v, CategoricalArray)
        # Use refs for CategoricalArray
        codes = refs(v)
        n_levels = length(levels(v))
    else
        # Convert plain vector to integer codes - handle mixed types
        if eltype(v) == Any
            # For Vector{Any}, extract all values and convert them consistently
            string_vals = String[string(x) for x in v]
            unique_vals = sort(unique(string_vals))
            code_map = Dict(val => i for (i, val) in enumerate(unique_vals))
            codes = [code_map[string(val)] for val in v]
        else
            # For homogeneous vectors
            unique_vals = sort(unique(v))
            code_map = Dict(val => i for (i, val) in enumerate(unique_vals))
            codes = [code_map[val] for val in v]
        end
        n_levels = length(unique_vals)
    end
    
    n_dummy_cols = size(M, 2)
    @assert n_dummy_cols == N "Contrast matrix produces $n_dummy_cols columns, expected $N"
    
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

function _evaluate_term_full!(term::InteractionTerm, data::NamedTuple, output::AbstractMatrix)
    n = size(output, 1)
    components = term.terms
    ncomponents = length(components)
    
    # Evaluate each component
    component_matrices = Vector{Matrix{Float64}}(undef, ncomponents)
    component_widths = Vector{Int}(undef, ncomponents)
    
    for (i, component) in enumerate(components)
        comp_width = width(component)
        component_widths[i] = comp_width
        component_matrices[i] = Matrix{Float64}(undef, n, comp_width)
        _evaluate_term_full!(component, data, component_matrices[i])
    end
    
    # Compute Kronecker product
    _compute_kronecker_product!(component_matrices, component_widths, output)
end

function _evaluate_term_full!(term::MatrixTerm, data::NamedTuple, output::AbstractMatrix)
    col_offset = 0
    
    for subterm in term.terms
        subterm_width = width(subterm)
        if subterm_width > 0
            subterm_output = view(output, :, col_offset+1:col_offset+subterm_width)
            _evaluate_term_full!(subterm, data, subterm_output)
            col_offset += subterm_width
        end
    end
end


function _evaluate_term_full!(term::ZScoredTerm, data::NamedTuple, output::AbstractMatrix)
    # Evaluate underlying term first
    _evaluate_term_full!(term.term, data, output)
    
    # Apply Z-score transformation to all columns
    _apply_zscore_transform!(output, term.center, term.scale)
    
    return output
end

# Single-column terms that need matrix interface
function _evaluate_term_full!(term::ContinuousTerm, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1
    copy!(view(output, :, 1), data[term.sym])
end

function _evaluate_term_full!(term::Term, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1
    copy!(view(output, :, 1), data[term.sym])
end

function _evaluate_term_full!(term::ConstantTerm, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1
    fill!(view(output, :, 1), term.n)
end

function _evaluate_term_full!(term::InterceptTerm{true}, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1
    fill!(view(output, :, 1), 1.0)
end

function _evaluate_term_full!(term::FunctionTerm, data::NamedTuple, output::AbstractMatrix)
    @assert size(output, 2) == 1
    _evaluate_single_column_direct!(term, data, view(output, :, 1))
end

###############################################################################
# Helper Functions
###############################################################################

function _compute_kronecker_product!(components::Vector{Matrix{Float64}}, widths::Vector{Int}, output::AbstractMatrix)
    n = size(output, 1)
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
end

"""
    _apply_zscore_single_column!(output::AbstractVector, center, scale)

Apply Z-score transformation to a single column vector in-place.
"""
function _apply_zscore_single_column!(output::AbstractVector, center, scale)
    if center isa Number && scale isa Number
        if center == 0
            inv_scale = 1.0 / scale
            @inbounds @simd for i in 1:length(output)
                output[i] *= inv_scale
            end
        else
            inv_scale = 1.0 / scale
            @inbounds @simd for i in 1:length(output)
                output[i] = (output[i] - center) * inv_scale
            end
        end
    elseif center isa AbstractVector && scale isa AbstractVector
        # This shouldn't happen for single column, but handle gracefully
        @assert length(center) == 1 && length(scale) == 1 "Vector center/scale for single column must have length 1"
        c, s = center[1], scale[1]
        if c == 0
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(output)
                output[i] *= inv_s
            end
        else
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(output)
                output[i] = (output[i] - c) * inv_s
            end
        end
    elseif center isa Number && scale isa AbstractVector
        @assert length(scale) == 1 "Vector scale for single column must have length 1"
        s = scale[1]
        if center == 0
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(output)
                output[i] *= inv_s
            end
        else
            inv_s = 1.0 / s
            @inbounds @simd for i in 1:length(output)
                output[i] = (output[i] - center) * inv_s
            end
        end
    elseif center isa AbstractVector && scale isa Number
        @assert length(center) == 1 "Vector center for single column must have length 1"
        c = center[1]
        if c == 0
            inv_scale = 1.0 / scale
            @inbounds @simd for i in 1:length(output)
                output[i] *= inv_scale
            end
        else
            inv_scale = 1.0 / scale
            @inbounds @simd for i in 1:length(output)
                output[i] = (output[i] - c) * inv_scale
            end
        end
    else
        error("Unsupported center/scale types for Z-score: $(typeof(center)), $(typeof(scale))")
    end
    
    return output
end

"""
    _apply_zscore_transform!(output::AbstractMatrix, center, scale)

Apply Z-score transformation to a matrix in-place, handling various center/scale combinations.
"""
function _apply_zscore_transform!(output::AbstractMatrix, center, scale)
    n_rows, n_cols = size(output)
    
    if center isa Number && scale isa Number
        # Scalar center and scale - apply to all columns
        if center == 0
            inv_scale = 1.0 / scale
            @inbounds for i in eachindex(output)
                output[i] *= inv_scale
            end
        else
            inv_scale = 1.0 / scale
            @inbounds for i in eachindex(output)
                output[i] = (output[i] - center) * inv_scale
            end
        end
    elseif center isa AbstractVector && scale isa AbstractVector
        # Vector center and scale - apply column-wise
        @assert length(center) == n_cols "Center vector length must match number of columns"
        @assert length(scale) == n_cols "Scale vector length must match number of columns"
        
        @inbounds for col in 1:n_cols
            c = center[col]
            s = scale[col]
            if c == 0
                inv_s = 1.0 / s
                for row in 1:n_rows
                    output[row, col] *= inv_s
                end
            else
                inv_s = 1.0 / s
                for row in 1:n_rows
                    output[row, col] = (output[row, col] - c) * inv_s
                end
            end
        end
    elseif center isa Number && scale isa AbstractVector
        # Scalar center, vector scale
        @assert length(scale) == n_cols "Scale vector length must match number of columns"
        
        @inbounds for col in 1:n_cols
            s = scale[col]
            if center == 0
                inv_s = 1.0 / s
                for row in 1:n_rows
                    output[row, col] *= inv_s
                end
            else
                inv_s = 1.0 / s
                for row in 1:n_rows
                    output[row, col] = (output[row, col] - center) * inv_s
                end
            end
        end
    elseif center isa AbstractVector && scale isa Number
        # Vector center, scalar scale
        @assert length(center) == n_cols "Center vector length must match number of columns"
        
        inv_scale = 1.0 / scale
        @inbounds for col in 1:n_cols
            c = center[col]
            if c == 0
                for row in 1:n_rows
                    output[row, col] *= inv_scale
                end
            else
                for row in 1:n_rows
                    output[row, col] = (output[row, col] - c) * inv_scale
                end
            end
        end
    else
        error("Unsupported center/scale types for Z-score: $(typeof(center)), $(typeof(scale))")
    end
    
    return output
end

###############################################################################
# Integration Functions
###############################################################################

"""
    build_enhanced_mapping(model::StatisticalModel) -> ColumnMapping

Create ColumnMapping with validation against fitted model.
"""
function build_enhanced_mapping(model::StatisticalModel)
    rhs = fixed_effects_form(model).rhs
    mapping = build_column_mapping(rhs, model)
    
    # Validate against actual model matrix if available
    try
        if hasmethod(modelmatrix, (typeof(model),))
            X_fitted = modelmatrix(model)
            if mapping.total_columns != size(X_fitted, 2)
                @warn "Column mapping mismatch: mapping has $(mapping.total_columns) columns, model matrix has $(size(X_fitted, 2)) columns"
            end
        end
    catch
        @warn "Could not validate mapping against model matrix"
    end
    
    return mapping
end

# Add alias for compatibility
enhanced_column_mapping = build_enhanced_mapping
