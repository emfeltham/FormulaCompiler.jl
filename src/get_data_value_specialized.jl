# get_data_value_specialized.jl

"""
    get_data_value_specialized(data::NamedTuple, column::Symbol, row_idx::Int) -> Any

Fetch a raw value from `data` by column symbol in a type-stable way.

# Arguments
- `data::NamedTuple`: Named-tuple of column arrays (e.g. `data.x`, `data.y`, ...).
- `column::Symbol`: The field name to extract (e.g. `:x`, `:y`, or any other column symbol).
- `row_idx::Int`: The index of the row to retrieve.

# Returns
The element `data[column][row_idx]`, with fast-paths for common symbols.

# Example
```julia
val = get_data_value_specialized(data, :x, 5)  # equivalent to data.x[5]
```

NOTE: this is a very, very stupid function. But, there may be a better solution
for no-allocation indexing??
"""
@inline function get_data_value_specialized(
    data::NamedTuple,
    column::Symbol,
    row_idx::Int
)
    if column === :x
        return data.x[row_idx]
    elseif column === :y
        return data.y[row_idx]
    elseif column === :z
        return data.z[row_idx]
    elseif column === :w
        return data.w[row_idx]
    elseif column === :t
        return data.t[row_idx]
    else
        # Fallback for any other column
        return data[column][row_idx]
    end
end