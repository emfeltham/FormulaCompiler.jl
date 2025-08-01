# kronecker.jl

###############################################################################
# KRONECKER PATTERN COMPUTATION
###############################################################################

"""
    compute_generalized_kronecker_pattern(component_widths::Vector{Int}) -> Vector{Vector{Int}}

Compute full Kronecker pattern for any N-way interaction. Always precomputes everything.
"""
function compute_generalized_kronecker_pattern(component_widths::Vector{Int})
    N = length(component_widths)
    
    if N == 0
        return Vector{Vector{Int}}[]
    elseif N == 1
        # Single component - trivial pattern
        return [[i] for i in 1:component_widths[1]]
    end
    
    total_terms = prod(component_widths)
    
    # Pre-allocate the full pattern
    pattern = Vector{Vector{Int}}(undef, total_terms)
    
    # Generate all combinations using Cartesian indices
    ranges = Tuple(1:w for w in component_widths)
    
    idx = 1
    for combo in Iterators.product(ranges...)
        pattern[idx] = collect(combo)  # Convert tuple to vector
        idx += 1
    end
    
    return pattern
end

"""
    compute_kronecker_pattern(component_widths::Vector{Int}) -> Vector{Tuple{Vararg{Int}}}

Compute the Kronecker-product index pattern for an interaction term of arbitrary arity.

# Arguments

- `component_widths::Vector{Int}`: A vector of positive integers, where each entry `w_i` is the number of columns contributed by the i‑th component (e.g., the number of basis functions, dummy columns, or features for that variable or transform).

# Returns

- `pattern::Vector{Tuple{Vararg{Int}}}`: A vector of tuples, each of length `n = length(component_widths)`. Each tuple `(i₁, i₂, …, iₙ)` represents one combination of column indices: take column `i₁` from component 1, `i₂` from component 2, …, `iₙ` from component n, and multiply them to form one column of the full interaction design.

# Details

- Preallocates a vector of length `prod(component_widths)` to hold all index combinations.
- Uses `Iterators.product` to efficiently generate the Cartesian product of the ranges `1:w_i`.
- Converts each `NTuple{n,Int}` returned by `product` into a plain `Tuple{Vararg{Int}}`.
- Complexity: Time and memory are O(∏₁ⁿ w_i). Suitable for moderate‑sized interactions.
- Throws an `ArgumentError` if any width is less than 1.

# Examples

```julia
julia> compute_kronecker_pattern([2, 3])
6-element Vector{Tuple{Vararg{Int}}}:
 (1, 1)
 (1, 2)
 (1, 3)
 (2, 1)
 (2, 2)
 (2, 3)

julia> compute_kronecker_pattern([2, 3, 2])
12-element Vector{Tuple{Vararg{Int}}}:
 (1, 1, 1)
 (1, 1, 2)
 (1, 2, 1)
 (1, 2, 2)
 (1, 3, 1)
 (1, 3, 2)
 (2, 1, 1)
 (2, 1, 2)
 (2, 2, 1)
 (2, 2, 2)
 (2, 3, 1)
 (2, 3, 2)
```
"""
function compute_kronecker_pattern(component_widths::Vector{Int})
    # Validate input
    if any(w -> w < 1, component_widths)
        throw(ArgumentError("All component widths must be positive integers. Received: $(component_widths)"))
    end

    N = length(component_widths)
    
    # Prepare ranges 1:w for each component - use ntuple for type stability
    ranges = ntuple(i -> 1:component_widths[i], N)

    # Preallocate output vector with parametric type
    total = prod(component_widths)
    pattern = Vector{NTuple{N,Int}}(undef, total)

    # Fill with index tuples from Cartesian product
    idx = 1
    for combo in Iterators.product(ranges...)
        pattern[idx] = combo  # combo is already NTuple{N,Int}
        idx += 1
    end

    return pattern
end

###############################################################################
# KRONECKER PATTERN APPLICATION
###############################################################################

"""
    apply_kronecker_pattern_to_positions!(
        pattern::Vector{NTuple{N,Int}},
        component_scratch_map::Vector{UnitRange{Int}},
        scratch::Vector{Float64},
        output::AbstractVector{Float64},
        output_positions::Vector{Int}
    ) where N

UPDATED: Apply Kronecker pattern to specific positions without enumerate().
Overwrites old method.
"""
function apply_kronecker_pattern_to_positions!(
    pattern::Vector{NTuple{N,Int}},
    component_scratch_map::Vector{UnitRange{Int}},
    scratch::Vector{Float64},
    output::AbstractVector{Float64},
    output_positions::Vector{Int}
) where N
        
    @inbounds for idx in 1:length(pattern)
        if idx <= length(output_positions)
            indices = pattern[idx]
            
            # Type-stable computation with compile-time known N
            product = 1.0
            for i in 1:N
                scratch_pos = first(component_scratch_map[i]) + indices[i] - 1
                product *= scratch[scratch_pos]
            end
            
            output[output_positions[idx]] = product
        end
    end
    
    return nothing
end

"""
    apply_kronecker_to_scratch_range!(
        pattern::Vector{NTuple{N,Int}},
        component_scratch_map::Vector{UnitRange{Int}},
        scratch::Vector{Float64},
        output_start::Int,
        output_end::Int
    ) where N

Build all N-way interaction products (“Kronecker terms”) by multiplying
component values already staged in `scratch`, writing the results into
`scratch[output_start:output_end]`.

# Arguments

* `pattern::Vector{NTuple{N,Int}}`
  Precomputed list of index tuples `(i₁, i₂, …, iₙ)`, one per interaction term.
* `maps::Vector{UnitRange{Int}}`
  For each of the N components, the `scratch` subrange in which that component’s
  values live.
* `scratch::Vector{Float64}`
  Working buffer containing component outputs.
* `output_start::Int`, `output_end::Int`
  Inclusive slice of `scratch` to populate with each product.

# Behavior

For each tuple in `pattern`, multiply the N component values:

```julia
prod = ∏_{j=1}^N scratch[first(maps[j]) + inds[j] - 1]
```

and assign

```julia
scratch[output_start + idx - 1] = prod
```

where `idx` marches from 1 to `length(pattern)`.

Returns `nothing`.  Fully in-place and allocation-free.

# Example

```julia
# If pattern = [(1,1),(1,2),(2,1),(2,2)]
# and two components each with scratch ranges  1:2 and 3:4,
# then this will compute  x1*x3, x1*x4, x2*x3, x2*x4  into scratch[5:8].
```

# Notes

N.B., `send` is unused

"""
function apply_kronecker_to_scratch_range!(
    pattern::Vector{NTuple{N,Int}},
    maps::Vector{UnitRange{Int}},
    scratch::Vector{Float64},
    sstart::Int,
    send::Int
) where N
    @inbounds for idx in 1:length(pattern)
        prod = 1.0
        inds = pattern[idx]
        # no view: compute each component by direct indexing
        for j in 1:N
            r = maps[j]
            prod *= scratch[first(r) + inds[j] - 1]
        end
        scratch[sstart + idx - 1] = prod
    end
end
