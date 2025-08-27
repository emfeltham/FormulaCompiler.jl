# overrides.jl - Override vector implementations for variable substitution

"""
Single-row override vector: returns replacement at `row`, base elsewhere (unused)
NOTE: Current override wrapper uses eltype Any; this impacts AD typing.
TODO(derivatives): Introduce `SingleRowOverrideVector{T}` and per-eltype merged data_over
to ensure getindex returns T (Float64/Dual), removing Any from the hot path.
"""
mutable struct SingleRowOverrideVector <: AbstractVector{Any}
    base::Any
    row::Int
    replacement::Any
end

Base.size(v::SingleRowOverrideVector) = size(v.base)
Base.length(v::SingleRowOverrideVector) = length(v.base)
Base.IndexStyle(::Type{<:SingleRowOverrideVector}) = IndexLinear()
Base.eltype(::Type{SingleRowOverrideVector}) = Any
Base.getindex(v::SingleRowOverrideVector, i::Int) = (i == v.row ? v.replacement : getindex(v.base, i))

"""
Typed single-row override vector that preserves element type T
"""
mutable struct TypedSingleRowOverrideVector{T} <: AbstractVector{T}
    base::AbstractVector
    row::Int
    replacement::T
end

Base.size(v::TypedSingleRowOverrideVector) = size(v.base)
Base.length(v::TypedSingleRowOverrideVector) = length(v.base)
Base.IndexStyle(::Type{<:TypedSingleRowOverrideVector}) = IndexLinear()
Base.eltype(::Type{TypedSingleRowOverrideVector{T}}) where {T} = T
@inline Base.getindex(v::TypedSingleRowOverrideVector{T}, i::Int) where {T} = (i == v.row ? v.replacement : convert(T, getindex(v.base, i)))

"""
Concrete Float64 override vector for FD evaluator (fully concrete base vector)
"""
mutable struct FDOverrideVector <: AbstractVector{Float64}
    base::Vector{Float64}
    row::Int
    replacement::Float64
end

Base.size(v::FDOverrideVector) = size(v.base)
Base.length(v::FDOverrideVector) = length(v.base)
Base.IndexStyle(::Type{<:FDOverrideVector}) = IndexLinear()
Base.eltype(::Type{FDOverrideVector}) = Float64
@inline Base.getindex(v::FDOverrideVector, i::Int) = (i == v.row ? v.replacement : v.base[i])

"""
    build_row_override_data(base::NamedTuple, vars::Vector{Symbol}, row::Int)

Build a NamedTuple overriding selected variables with SingleRowOverrideVector wrappers
"""
function build_row_override_data(base::NamedTuple, vars::Vector{Symbol}, row::Int)
    overrides = NamedTuple()
    override_vecs = Vector{SingleRowOverrideVector}(undef, length(vars))
    # Construct override vectors and merge into NamedTuple shadowing base
    pairs = Pair{Symbol,Any}[]
    for (i, s) in enumerate(vars)
        ov = SingleRowOverrideVector(getproperty(base, s), row, nothing)
        override_vecs[i] = ov
        push!(pairs, s => ov)
    end
    data_over = (; base..., pairs...)
    return data_over, override_vecs
end

"""
    build_row_override_data_typed(base::NamedTuple, vars::Vector{Symbol}, row::Int, ::Type{T})

Build a NamedTuple overriding selected variables using TypedSingleRowOverrideVector{T}
"""
function build_row_override_data_typed(base::NamedTuple, vars::Vector{Symbol}, row::Int, ::Type{T}) where {T}
    override_vecs = Vector{TypedSingleRowOverrideVector{T}}(undef, length(vars))
    pairs = Pair{Symbol,TypedSingleRowOverrideVector{T}}[]
    for (i, s) in enumerate(vars)
        col = getproperty(base, s)
        ov = TypedSingleRowOverrideVector{T}(col, row, zero(T))
        override_vecs[i] = ov
        push!(pairs, s => ov)
    end
    data_over = (; base..., pairs...)
    return data_over, override_vecs
end