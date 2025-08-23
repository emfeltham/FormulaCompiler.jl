# get_data_value_specialized.jl

"""
    get_data_value_specialized(data::NamedTuple, column::Symbol, row_idx::Int) -> Any

NOTE: this is a very, very stupid function. But, there may be a better solution
for no-allocation indexing?
"""
@inline function get_data_value_specialized(
    data::NamedTuple,
    column::Symbol,
    row_idx::Int
)
    return data[column][row_idx]
end

"""
    get_data_value_type_stable(data::NamedTuple, ::Val{column}, row_idx::Int)

Type-stable accessor using Val{column} to enable compile-time dispatch and
avoid dynamic symbol lookup. Returns the element at `row_idx` for the given column.
"""
@inline function get_data_value_type_stable(
    data::NamedTuple{names},
    ::Val{column},
    row_idx::Int,
) where {names, column}
    # Compute field index at compile-time from NamedTuple type and column symbol
    idx = Base.fieldindex(NamedTuple{names}, column)
    return getfield(data, idx)[row_idx]
end
