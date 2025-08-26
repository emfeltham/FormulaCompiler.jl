# core/utilities.jl
# Core utility functions and types used throughout the system

# useful for booleans in formulas
not(x::Bool) = !x
# N.B., this is dangerous -- does not clearly fail when x outside [0,1]
not(x::T) where {T<:Real} = one(x) - x

"""
    OverrideVector{T} <: AbstractVector{T}

A lazy vector that returns the same override value for all indices.
This avoids allocating full arrays when setting all observations to a representative value.

# Example
```julia
# Instead of: fill(2.5, 1_000_000)  # Allocates 8MB
# Use: OverrideVector(2.5, 1_000_000)  # Allocates ~32 bytes
```
"""
struct OverrideVector{T} <: AbstractVector{T}
    override_value::T
    length::Int
    
    function OverrideVector(value::T, length::Int) where T
        new{T}(value, length)
    end
end