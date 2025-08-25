# UnifiedCompiler Scratch Space Management
# Simple pre-allocated scratch array for zero-allocation execution

# Single scratch space with size-based access
mutable struct ScratchPool
    small::Vector{Float64}   # Up to 100 elements
    medium::Vector{Float64}  # Up to 1000 elements
    large::Vector{Float64}   # Up to 10000 elements
    xlarge::Vector{Float64}  # Dynamically sized
    
    ScratchPool() = new(
        Vector{Float64}(undef, 100),
        Vector{Float64}(undef, 1000),
        Vector{Float64}(undef, 10000),
        Vector{Float64}()
    )
end

const SCRATCH_POOL = ScratchPool()

function get_scratch_for_size(size::Int)
    if size <= 100
        return view(SCRATCH_POOL.small, 1:size)
    elseif size <= 1000
        return view(SCRATCH_POOL.medium, 1:size)
    elseif size <= 10000
        return view(SCRATCH_POOL.large, 1:size)
    else
        # Resize xlarge if needed
        if length(SCRATCH_POOL.xlarge) < size
            resize!(SCRATCH_POOL.xlarge, size)
        end
        return view(SCRATCH_POOL.xlarge, 1:size)
    end
end

# Compile-time scratch size dispatch
@inline get_scratch(::Val{N}) where N = get_scratch_for_size(N)

# Clear scratch space (for safety/debugging)
function clear_scratch!(scratch::AbstractVector{Float64})
    fill!(scratch, 0.0)
end

# Pre-warm scratch pools (optional, for benchmarking)
function warm_scratch_pools(max_size::Int = 1000)
    scratch = get_scratch_for_size(max_size)
    clear_scratch!(scratch)
end