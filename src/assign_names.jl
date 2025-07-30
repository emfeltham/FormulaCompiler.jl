# execution_plans.jl

###############################################################################
# Assign unique names
###############################################################################

const VAR_COUNTER = Ref(0)

function next_var(prefix::String="v")
    VAR_COUNTER[] += 1
    return "$(prefix)_$(VAR_COUNTER[])"
end

function reset_var_counter!()
    VAR_COUNTER[] = 0
end
