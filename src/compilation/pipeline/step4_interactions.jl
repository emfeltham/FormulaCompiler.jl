# step4_interactions.jl
# Main entry point for the interaction system (Step 4 of compilation pipeline)
# 
# This file provides a clean interface while organizing the complex interaction system
# into logical components within the step4/ subdirectory.

# Type definitions
include("step4/types.jl")

# Main implementation (will be further split in future)
include("step4/main.jl")