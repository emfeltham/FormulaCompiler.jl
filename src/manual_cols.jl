# Manual Dispatch _cols! - Eliminate Dynamic Dispatch

###############################################################################
# Manual Type Dispatch - Avoid AbstractTerm Dynamic Dispatch
###############################################################################

function _cols!(row_vec, terms::Vector{AbstractTerm}, data, pos=1)
    current_pos = pos
    
    @inbounds for i in 1:length(terms)
        term = terms[i]
        
        # Manual dispatch on concrete types - no dynamic method lookup
        if term isa ContinuousTerm
            row_vec[current_pos] = Float64(data[term.sym][1])
            current_pos += 1
            
        elseif term isa Term
            row_vec[current_pos] = Float64(data[term.sym][1])
            current_pos += 1
            
        elseif term isa ConstantTerm
            row_vec[current_pos] = Float64(term.n)
            current_pos += 1
            
        elseif term isa InterceptTerm{true}
            row_vec[current_pos] = 1.0
            current_pos += 1
            
        elseif term isa InterceptTerm{false}
            # Skip - produces no columns
            
        elseif term isa CategoricalTerm
            # Inline categorical processing to avoid function call overhead
            v = data[term.sym][1]
            code = v isa CategoricalValue ? levelcode(v) : 1
            M = term.contrasts.matrix
            w = size(M, 2)
            
            for k in 1:w
                row_vec[current_pos + k - 1] = Float64(M[code, k])
            end
            current_pos += w
            
        elseif term isa FunctionTerm
            # Use existing function - it's already optimized
            current_pos = _cols_function!(row_vec, term, data, current_pos)
            
        elseif term isa InteractionTerm
            # Use existing function - it's already optimized  
            current_pos = _cols_interaction!(row_vec, term, data, current_pos)
            
        elseif term isa ZScoredTerm
            # Inline Z-score processing
            val = _get_scalar_value_inline(term.term, data)
            center = term.center isa Number ? term.center : term.center[1]
            scale = term.scale isa Number ? term.scale : term.scale[1]
            row_vec[current_pos] = (val - center) / scale
            current_pos += 1
            
        elseif term isa MatrixTerm
            # Recurse into matrix terms
            current_pos = _cols!(row_vec, term.terms, data, current_pos)
            
        else
            # Fallback - should rarely be hit
            @warn "Unknown term type: $(typeof(term))"
            current_pos += 1
        end
    end
    
    return current_pos
end

###############################################################################
# Specialized Helper Functions - Avoid Generic Dispatch
###############################################################################

@inline function _get_scalar_value_inline(term::ContinuousTerm, data)
    return Float64(data[term.sym][1])
end

@inline function _get_scalar_value_inline(term::Term, data)
    return Float64(data[term.sym][1])
end

@inline function _get_scalar_value_inline(term::ConstantTerm, data)
    return Float64(term.n)
end

@inline function _get_scalar_value_inline(term::InterceptTerm{true}, data)
    return 1.0
end

@inline function _get_scalar_value_inline(term::InterceptTerm{false}, data)
    return 0.0
end

@inline function _get_scalar_value_inline(term::CategoricalTerm, data)
    v = data[term.sym][1]
    code = v isa CategoricalValue ? levelcode(v) : 1
    return Float64(term.contrasts.matrix[code, 1])
end

# Specialized function term processor
function _cols_function!(row_vec, term::FunctionTerm, data, pos)
    nargs = length(term.args)
    
    result = if nargs == 1
        arg = _get_scalar_value_inline(term.args[1], data)
        _safe_function_call_inline(term.f, arg)
    elseif nargs == 2
        arg1 = _get_scalar_value_inline(term.args[1], data)
        arg2 = term.args[2] isa ConstantTerm ? Float64(term.args[2].n) : _get_scalar_value_inline(term.args[2], data)
        _safe_function_call_2_inline(term.f, arg1, arg2)
    else
        1.0  # Fallback
    end
    
    @inbounds row_vec[pos] = Float64(result)
    return pos + 1
end

# Specialized interaction term processor
function _cols_interaction!(row_vec, term::InteractionTerm, data, pos)
    w = width(term)
    components = term.terms
    
    if w == 1
        # Simple product
        product = 1.0
        for comp in components
            product *= _get_scalar_value_inline(comp, data)
        end
        @inbounds row_vec[pos] = product
        return pos + 1
    else
        # Multi-column - use optimized Kronecker
        _fill_kronecker_inline!(row_vec, components, data, pos, w)
        return pos + w
    end
end

###############################################################################
# Inlined Safe Function Calls
###############################################################################

@inline function _safe_function_call_inline(f, arg)
    if f === log
        return arg > 0 ? log(arg) : arg
    elseif f === exp
        return exp(arg)
    elseif f === sqrt
        return arg >= 0 ? sqrt(arg) : arg
    elseif f === abs
        return abs(arg)
    else
        try
            return Float64(f(arg))
        catch
            return Float64(arg)
        end
    end
end

@inline function _safe_function_call_2_inline(f, arg1, arg2)
    if f === (+)
        return arg1 + arg2
    elseif f === (-)
        return arg1 - arg2
    elseif f === (*)
        return arg1 * arg2
    elseif f === (/)
        return arg2 != 0 ? arg1 / arg2 : arg1
    elseif f === (^)
        return arg1 ^ arg2
    else
        try
            return Float64(f(arg1, arg2))
        catch
            return Float64(arg1)
        end
    end
end

###############################################################################
# Inlined Kronecker Product
###############################################################################

function _fill_kronecker_inline!(row_vec, components, data, pos, total_width)
    n_components = length(components)
    
    if n_components == 2
        # Two-component case - most common
        comp1, comp2 = components[1], components[2]
        w1, w2 = width(comp1), width(comp2)
        
        if w1 == 1 && w2 == 1
            val1 = _get_scalar_value_inline(comp1, data)
            val2 = _get_scalar_value_inline(comp2, data)
            @inbounds row_vec[pos] = val1 * val2
        elseif w1 == 1
            val1 = _get_scalar_value_inline(comp1, data)
            v2 = data[comp2.sym][1]
            code2 = v2 isa CategoricalValue ? levelcode(v2) : 1
            M2 = comp2.contrasts.matrix
            for j in 1:w2
                @inbounds row_vec[pos + j - 1] = val1 * Float64(M2[code2, j])
            end
        elseif w2 == 1
            val2 = _get_scalar_value_inline(comp2, data)
            v1 = data[comp1.sym][1]
            code1 = v1 isa CategoricalValue ? levelcode(v1) : 1
            M1 = comp1.contrasts.matrix
            for i in 1:w1
                @inbounds row_vec[pos + i - 1] = Float64(M1[code1, i]) * val2
            end
        else
            # Both multi-column
            v1 = data[comp1.sym][1]
            v2 = data[comp2.sym][1]
            code1 = v1 isa CategoricalValue ? levelcode(v1) : 1
            code2 = v2 isa CategoricalValue ? levelcode(v2) : 1
            M1 = comp1.contrasts.matrix
            M2 = comp2.contrasts.matrix
            
            idx = 0
            for i in 1:w1
                val1 = Float64(M1[code1, i])
                for j in 1:w2
                    val2 = Float64(M2[code2, j])
                    @inbounds row_vec[pos + idx] = val1 * val2
                    idx += 1
                end
            end
        end
    else
        # General case for 3+ components
        @inbounds for out_idx in 1:total_width
            product = 1.0
            temp_idx = out_idx - 1
            
            for comp_idx in n_components:-1:1
                comp = components[comp_idx]
                w = width(comp)
                local_idx = temp_idx % w + 1
                temp_idx ÷= w
                
                val = if w == 1
                    _get_scalar_value_inline(comp, data)
                else
                    v = data[comp.sym][1]
                    code = v isa CategoricalValue ? levelcode(v) : 1
                    Float64(comp.contrasts.matrix[code, local_idx])
                end
                
                product *= val
            end
            
            row_vec[pos + out_idx - 1] = product
        end
    end
end