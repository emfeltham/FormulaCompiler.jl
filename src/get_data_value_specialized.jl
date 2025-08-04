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
