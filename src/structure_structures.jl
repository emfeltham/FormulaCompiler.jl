# structure_structures.jl

###############################################################################
# Term Analysis Types
###############################################################################

"""
Analysis result for a single term, containing everything needed for compilation.
"""
struct TermAnalysis
    term::AbstractTerm                    # Original StatsModels term
    start_position::Int                   # First column in row_vec
    width::Int                           # Number of columns this term produces
    columns_used::Vector{Symbol}         # Data columns this term reads
    term_type::Symbol                    # :constant, :continuous, :categorical, :function, :interaction, :zscore
    metadata::Dict{Symbol, Any}          # Type-specific information
end

"""
Complete analysis of a formula's structure.
"""
struct FormulaAnalysis
    terms::Vector{TermAnalysis}          # All terms in evaluation order
    total_width::Int                     # Total columns in model matrix
    all_columns::Vector{Symbol}          # All data columns used
    position_map::Dict{AbstractTerm, UnitRange{Int}}  # Term -> column range mapping
end