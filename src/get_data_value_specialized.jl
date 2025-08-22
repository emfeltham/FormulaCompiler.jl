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

Type-stable version using Val{symbol} for compile-time dispatch.
"""
@inline function get_data_value_type_stable(
    data::NamedTuple,
    ::Val{column},
    row_idx::Int
) where column
    return data[column][row_idx]
end
