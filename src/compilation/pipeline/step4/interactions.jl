# step4/interactions.jl  
# Main entry point for step4 interactions system
# Complete interaction system with zero allocations for all cases (except functions!)

# Load all components in dependency order
include("types.jl")

# Load the rest from the original file for now (we can split further later)
# This preserves functionality while providing some organization
