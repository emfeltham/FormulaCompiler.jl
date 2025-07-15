# Validation helpers
# OLD -- REPURPOSE MAYBE

function validate_data_compatibility(ipm::InplaceModeler, data::NamedTuple)
    
    # Not sure if I want to restrict to same length!
    # nrows = length(first(data))
    # expected_rows = size(first(ipm.fn_scratch), 1)  # or track this separately
    
    # if nrows != expected_rows
    #     throw(ArgumentError("Data has $nrows rows, but InplaceModeler was created for $expected_rows rows"))
    # end
    
    # Check that all required columns exist
    rhs = formula(ipm.model).rhs
    required_vars = Set(termvars(rhs))  # You'd need to implement termvars
    available_vars = Set(keys(data))
    
    missing_vars = setdiff(required_vars, available_vars)
    if !isempty(missing_vars)
        throw(ArgumentError("Missing required variables: $(collect(missing_vars))"))
    end
end
