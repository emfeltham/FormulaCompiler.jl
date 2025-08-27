# contrasts.jl - Discrete contrasts for categorical variables

"""
    contrast_modelrow!(Δ, compiled, data, row; var, from, to)

Compute a discrete contrast at one row for a single variable: `Δ = X(to) − X(from)`.

Arguments:
- `Δ::AbstractVector{Float64}`: Preallocated buffer of length `n_terms`.
- `compiled::UnifiedCompiled`: Result of `compile_formula`.
- `data::NamedTuple`: Column-table data.
- `row::Int`: Row index.
- `var::Symbol`: Variable to change (e.g., `:group3`).
- `from`, `to`: Values to contrast (level names or `CategoricalValue` for categorical; numbers for discrete).

Notes:
- Uses a row-local override; for categorical columns, values are normalized to the column's levels.
"""
function contrast_modelrow!(
    Δ::AbstractVector{Float64},
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    row::Int;
    var::Symbol,
    from,
    to,
) where {T, Ops, S, O}
    @assert length(Δ) == length(compiled)
    # Build override wrapper for just this variable
    data_over, overrides = build_row_override_data(data, [var], row)
    # If categorical, ensure replacement is a CategoricalValue consistent with column levels
    base_col = getproperty(data, var)
    y_from = Vector{Float64}(undef, length(compiled))
    y_to = Vector{Float64}(undef, length(compiled))
    # From
    if (Base.find_package("CategoricalArrays") !== nothing) && (base_col isa CategoricalArrays.CategoricalArray)
        levels_list = levels(base_col)
        temp = CategoricalArrays.categorical([from], levels=levels_list, ordered=CategoricalArrays.isordered(base_col))
        overrides[1].replacement = temp[1]
    else
        overrides[1].replacement = from
    end
    compiled(y_from, data_over, row)
    # To
    if (Base.find_package("CategoricalArrays") !== nothing) && (base_col isa CategoricalArrays.CategoricalArray)
        levels_list = levels(base_col)
        temp = CategoricalArrays.categorical([to], levels=levels_list, ordered=CategoricalArrays.isordered(base_col))
        overrides[1].replacement = temp[1]
    else
        overrides[1].replacement = to
    end
    compiled(y_to, data_over, row)
    # Δ
    @inbounds @fastmath for i in 1:length(compiled)
        Δ[i] = y_to[i] - y_from[i]
    end
    return Δ
end

function contrast_modelrow(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    row::Int;
    var::Symbol,
    from,
    to,
) where {T, Ops, S, O}
    Δ = Vector{Float64}(undef, length(compiled))
    contrast_modelrow!(Δ, compiled, data, row; var=var, from=from, to=to)
    return Δ
end