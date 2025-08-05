# apply_function.jl

###############################################################################
# APPLY FUNCTIONS
###############################################################################

"""
    apply_function_direct_single(func::Function, val::Float64) -> Float64

Apply single-argument function directly with domain checking.
No varargs overhead, concrete Float64 type.
"""
function apply_function_direct_single(func::Function, val::Float64)
    if func === log
        return val > 0.0 ? log(val) : (val == 0.0 ? -Inf : NaN)
    elseif func === exp
        return exp(clamp(val, -700.0, 700.0))  # Prevent overflow
    elseif func === sqrt
        return val ≥ 0.0 ? sqrt(val) : NaN
    elseif func === abs
        return abs(val)
    elseif func === sin
        return sin(val)
    elseif func === cos
        return cos(val)
    elseif func === tan
        return tan(val)
    else
        # Direct function call for other functions
        return Float64(func(val))
    end
end

"""
    apply_function_direct_binary(func::Function, val1::Float64, val2::Float64) -> Float64

Apply binary function directly with domain checking.
No varargs overhead, concrete Float64 types.
"""
function apply_function_direct_binary(func::Function, val1::Float64, val2::Float64)
    if func === (+)
        return val1 + val2
    elseif func === (-)
        return val1 - val2
    elseif func === (*)
        return val1 * val2
    elseif func === (/)
        return val2 == 0.0 ? (val1 == 0.0 ? NaN : (val1 > 0.0 ? Inf : -Inf)) : val1 / val2
    elseif func === (^)
        if val1 == 0.0 && val2 < 0.0
            return Inf
        elseif val1 < 0.0 && !isinteger(val2)
            return NaN
        else
            return val1^val2
        end
    else
        return Float64(func(val1, val2))
    end
end
