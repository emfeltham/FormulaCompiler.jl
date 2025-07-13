# InplaceModeler.jl - RIGHT-SIZED VERSION: Only allocate scratch for terms that generate columns

#───────────────────────────────────────────────────────────────────────────────
# Simple filtering using existing mapping infrastructure
#───────────────────────────────────────────────────────────────────────────────

"""
    extract_active_function_terms(mapping::ColumnMapping) -> Vector{FunctionTerm}

Extract FunctionTerms that actually generate columns from the existing mapping.
The mapping already knows which terms generate columns!
"""
function extract_active_function_terms(mapping::ColumnMapping)
    active_fn_terms = FunctionTerm[]
    
    # The mapping already has all terms that generate columns
    for (term, range) in mapping.term_info
        if !isempty(range)
            if term isa FunctionTerm
                push!(active_fn_terms, term)
            elseif term isa InteractionTerm
                # Check components for FunctionTerms
                for component in term.terms
                    if component isa FunctionTerm && component ∉ active_fn_terms
                        push!(active_fn_terms, component)
                    end
                end
            end
        end
    end
    
    return active_fn_terms
end

"""
    extract_active_interaction_terms(mapping::ColumnMapping) -> Vector{InteractionTerm}

Extract InteractionTerms that actually generate columns from the existing mapping.
"""
function extract_active_interaction_terms(mapping::ColumnMapping)
    active_int_terms = InteractionTerm[]
    
    for (term, range) in mapping.term_info
        if !isempty(range) && term isa InteractionTerm
            push!(active_int_terms, term)
        end
    end
    
    return active_int_terms
end

#───────────────────────────────────────────────────────────────────────────────
# 1.  State object with right-sized scratch buffer allocation
#───────────────────────────────────────────────────────────────────────────────

"""
    InplaceModeler{M}

A helper object that encapsulates a fitted `StatisticalModel` and
pre-allocated scratch buffers, but only for terms that actually generate columns.

# Key Change: Only allocates scratch for function/interaction terms that appear in the model matrix.
"""
mutable struct InplaceModeler{M}
    model       :: M
    
    # RIGHT-SIZED: Only store terms that actually generate columns
    active_fn_terms    :: Vector{FunctionTerm}           # Only FunctionTerms that generate columns
    fn_scratch         :: Vector{Matrix{Float64}}        # Scratch only for active terms
    fn_term_to_index   :: Dict{FunctionTerm, Int}        # Map term to scratch index
    
    active_int_terms   :: Vector{InteractionTerm}        # Only InteractionTerms that generate columns  
    int_subw           :: Vector{Vector{Int}}             # Component widths for active terms only
    int_stride         :: Vector{Vector{Int}}             # Strides for active terms only
    int_prefix         :: Vector{Vector{Int}}             # Prefixes for active terms only
    int_scratch        :: Vector{Matrix{Float64}}         # Scratch only for active terms
    int_term_to_index  :: Dict{InteractionTerm, Int}     # Map term to scratch index
end

"""
    filter_active_function_terms(mapping::ColumnMapping) -> Vector{FunctionTerm}

Extract FunctionTerms that actually generate columns from the existing mapping.
"""
function filter_active_function_terms(mapping::ColumnMapping)
    return extract_active_function_terms(mapping)
end

"""
    filter_active_interaction_terms(mapping::ColumnMapping) -> Vector{InteractionTerm}

Extract InteractionTerms that actually generate columns from the existing mapping.
"""
function filter_active_interaction_terms(mapping::ColumnMapping)
    return extract_active_interaction_terms(mapping)
end



"""
    InplaceModeler(model, nrows::Int) -> InplaceModeler{<:StatisticalModel}

RIGHT-SIZED: Create an InplaceModeler that only allocates scratch for terms that generate columns.
"""
function InplaceModeler(model, nrows::Int)
    rhs = fixed_effects_form(model).rhs
    
    # Build mapping to understand which terms generate columns
    mapping = build_column_mapping(rhs, model)
    
    # RIGHT-SIZED: Only find active terms that actually generate columns
    active_fn_terms = filter_active_function_terms(mapping)
    active_int_terms = filter_active_interaction_terms(mapping)
    
    # RIGHT-SIZED: Only allocate scratch for active function terms
    fn_scratch = [Matrix{Float64}(undef, nrows, length(ft.args)) for ft in active_fn_terms]
    fn_term_to_index = Dict(term => i for (i, term) in enumerate(active_fn_terms))
    
    # RIGHT-SIZED: Only allocate scratch for active interaction terms
    int_subw = [[width(p) for p in it.terms] for it in active_int_terms]
    int_prefix = [cumsum([0; sw[1:end-1]]) for sw in int_subw]
    int_stride = [let s = Vector{Int}(undef, length(sw))
                      s[1] = 1
                      for k in 2:length(sw)
                          s[k] = s[k-1] * sw[k-1]
                      end
                      s
                  end for sw in int_subw]
    int_scratch = [Matrix{Float64}(undef, nrows, sum(sw)) for sw in int_subw]
    int_term_to_index = Dict(term => i for (i, term) in enumerate(active_int_terms))

    return InplaceModeler{typeof(model)}(
        model,
        active_fn_terms, fn_scratch, fn_term_to_index,
        active_int_terms, int_subw, int_stride, int_prefix, int_scratch, int_term_to_index
    )
end

"""
    InplaceModeler(model::StatisticalModel, data::Tables.ColumnTable) -> InplaceModeler{<:StatisticalModel}

RIGHT-SIZED: Create an InplaceModeler with right-sized scratch allocation.
"""
function InplaceModeler(model::StatisticalModel, data::Tables.ColumnTable)
    nrows = Tables.istable(data) ? length(Tables.rows(data)) : length(first(data))
    InplaceModeler(model, nrows)
end

"""
    get_fn_scratch_index(ipm::InplaceModeler, term::FunctionTerm) -> Int

Get the scratch matrix index for a specific FunctionTerm.
Throws an error if the term is not active (doesn't generate columns).
"""
function get_fn_scratch_index(ipm::InplaceModeler, term::FunctionTerm)
    if haskey(imp.fn_term_to_index, term)
        return ipm.fn_term_to_index[term]
    else
        error("FunctionTerm $term is not active (doesn't generate columns). This suggests a bug in term filtering.")
    end
end

"""
    get_int_scratch_index(ipm::InplaceModeler, term::InteractionTerm) -> Int

Get the scratch matrix index for a specific InteractionTerm.
Throws an error if the term is not active (doesn't generate columns).
"""
function get_int_scratch_index(ipm::InplaceModeler, term::InteractionTerm)
    if haskey(ipm.int_term_to_index, term)
        return ipm.int_term_to_index[term]
    else
        error("InteractionTerm $term is not active (doesn't generate columns). This suggests a bug in term filtering.")
    end
end
