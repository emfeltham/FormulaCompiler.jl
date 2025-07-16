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
