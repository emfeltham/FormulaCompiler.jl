# =============================================================================
# CORE ABSTRACTION: CachedModelMatrix
# =============================================================================

"""
    TermDependencyCache

Precomputed dependency relationships between data columns and model matrix columns.
This is the key optimization - we analyze the formula structure once and reuse
the analysis for all subsequent updates.

# Fields
- `data_to_matrix_cols::Dict{Symbol, Vector{Int}}`: Maps data column → matrix column indices
- `matrix_col_to_term::Dict{Int, Tuple{Int, UnitRange{Int}}}`: Maps matrix column → (term_index, term_column_range)
- `terms::Vector{AbstractTerm}`: Original terms from the formula
- `term_ranges::Vector{UnitRange{Int}}`: Column ranges for each term in the full matrix
- `term_widths::Vector{Int}`: Cached widths to avoid recomputation
"""
struct TermDependencyCache
    data_to_matrix_cols::Dict{Symbol, Vector{Int}}
    matrix_col_to_term::Dict{Int, Tuple{Int, UnitRange{Int}}}
    terms::Vector{AbstractTerm}
    term_ranges::Vector{UnitRange{Int}}
    term_widths::Vector{Int}
    
    function TermDependencyCache(data_to_matrix_cols, matrix_col_to_term, 
                                terms, term_ranges, term_widths)
        new(data_to_matrix_cols, matrix_col_to_term, terms, term_ranges, term_widths)
    end
end

"""
    CachedModelMatrix{T <: AbstractMatrix}

A wrapper around a model matrix that enables efficient selective updates.
Uses cached dependency analysis to only recompute columns when their source
variables actually change.

# Fields
- `matrix::T`: The actual model matrix
- `cache::TermDependencyCache`: Cached dependency information
- `schema_rhs`: The schema-applied RHS for recomputation
"""
struct CachedModelMatrix{T <: AbstractMatrix}
    matrix::T
    cache::TermDependencyCache
    schema_rhs::AbstractTerm
    
    function CachedModelMatrix(matrix::T, cache::TermDependencyCache, 
                              schema_rhs::AbstractTerm) where T
        new{T}(matrix, cache, schema_rhs)
    end
end
