# core/utilities.jl
# Core utility functions and types used throughout the system

"""
    not(x)

Logical NOT operation for use in formula specifications.

# Arguments
- `x::Bool`: Returns the logical negation (!x)
- `x::Real`: Returns 1 - x (useful for probability complements)

# Returns
- For Bool: The opposite boolean value
- For Real: The complement (1 - x)

# Example
```julia
# In a formula
model = lm(@formula(y ~ not(treatment)), df)

# For probabilities
p = 0.3
q = not(p)  # 0.7
```

!!! warning
    For Real values, this assumes x is in [0,1] range. No bounds checking is performed.
"""
not(x::Bool) = !x
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