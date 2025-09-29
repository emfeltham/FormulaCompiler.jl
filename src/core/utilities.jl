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



# Mixture Detection Utilities
# Following Phase 1 implementation from CATEGORICAL_MIXTURES_DESIGN.md

"""
    is_mixture_column(col)

Detect if a column contains categorical mixture specifications.
Uses type-safe checking for CategoricalMixture objects.

# Example
```julia
# Returns true for mixture specifications
is_mixture_column([mix("A" => 0.3, "B" => 0.7), mix("A" => 0.3, "B" => 0.7)])

# Returns false for regular categorical data
is_mixture_column(["A", "B", "A"])
```
"""
function is_mixture_column(col)
    isempty(col) && return false
    first_element = col[1]
    
    # Check if column contains mixture objects by checking for expected properties
    # This will be overridden once CategoricalMixture is defined
    if !hasfield(typeof(first_element), :levels) || !hasfield(typeof(first_element), :weights)
        return false
    end
    
    # For mixture columns, all elements should have the same mixture specification
    # This is required for compile-time mixture support
    first_spec = (levels=first_element.levels, weights=first_element.weights)
    for element in col
        if !hasfield(typeof(element), :levels) || !hasfield(typeof(element), :weights)
            return false  # Not all elements are mixtures
        end
        spec = (levels=element.levels, weights=element.weights)
        if spec != first_spec
            return false  # Inconsistent mixture specifications
        end
    end
    
    return true
end

"""
    extract_mixture_spec(mixture_obj)

Extract the levels and weights from a mixture object.
Returns a NamedTuple with `levels` and `weights` fields.

# Example
```julia
mixture = mix("A" => 0.3, "B" => 0.7)
spec = extract_mixture_spec(mixture)
# spec.levels  # ["A", "B"]
# spec.weights # [0.3, 0.7]
```
"""
function extract_mixture_spec(mixture_obj)
    return (levels=mixture_obj.levels, weights=mixture_obj.weights)
end

"""
    validate_mixture_consistency!(data)

Validate that all mixture columns in the data have consistent specifications.
This is required for compile-time mixture support where all rows must have
identical mixture specifications.

# Arguments
- `data`: NamedTuple containing data columns

# Throws
- `ArgumentError`: If mixture columns have inconsistent specifications
- `ArgumentError`: If mixture weights don't sum to 1.0

# Example
```julia
# Valid data - all rows have same mixture
data = (x = [1.0, 2.0], group = [mix("A"=>0.3, "B"=>0.7), mix("A"=>0.3, "B"=>0.7)])
validate_mixture_consistency!(data)  # Passes

# Invalid data - inconsistent mixtures
data = (x = [1.0, 2.0], group = [mix("A"=>0.3, "B"=>0.7), mix("A"=>0.5, "B"=>0.5)])
validate_mixture_consistency!(data)  # Throws ArgumentError
```
"""
function validate_mixture_consistency!(data)
    for (col_name, col_data) in pairs(data)
        if is_mixture_column(col_data)
            validate_mixture_column!(col_name, col_data)
        end
    end
end

"""
    validate_mixture_column!(col_name, col_data)

Validate a single mixture column for consistency and correctness.

# Arguments
- `col_name`: Name of the column (for error messages)
- `col_data`: Vector containing mixture objects

# Throws
- `ArgumentError`: If mixtures are inconsistent or weights don't sum to 1.0
"""
function validate_mixture_column!(col_name, col_data)
    if isempty(col_data)
        return  # Empty columns are valid (though unusual)
    end
    
    # Check all rows have same mixture specification
    first_spec = extract_mixture_spec(col_data[1])
    for (i, row_mixture) in enumerate(col_data)
        if i == 1
            continue  # Skip first element (used as reference)
        end
        
        if !hasfield(typeof(row_mixture), :levels) || !hasfield(typeof(row_mixture), :weights)
            throw(ArgumentError("Inconsistent mixture specification in column $col_name at row $i: expected mixture object, got $(typeof(row_mixture))"))
        end
        
        spec = extract_mixture_spec(row_mixture)
        if spec != first_spec
            throw(ArgumentError("Inconsistent mixture specification in column $col_name at row $i: expected $(first_spec), got $(spec)"))
        end
    end
    
    # Validate weights sum to 1.0
    if !isapprox(sum(first_spec.weights), 1.0, atol=1e-10)
        throw(ArgumentError("Mixture weights in column $col_name do not sum to 1.0: $(first_spec.weights) (sum = $(sum(first_spec.weights)))"))
    end
    
    # Validate no duplicate levels
    if length(unique(first_spec.levels)) != length(first_spec.levels)
        throw(ArgumentError("Mixture in column $col_name contains duplicate levels: $(first_spec.levels)"))
    end
    
    # Validate all weights are non-negative
    if any(w < 0 for w in first_spec.weights)
        throw(ArgumentError("Mixture weights in column $col_name must be non-negative: $(first_spec.weights)"))
    end
end

# Helper Functions for Mixture Creation (Phase 4)

"""
    create_mixture_column(mixture_spec, n_rows::Int)

Create a column of identical mixture specifications for use in reference grids.

# Arguments
- `mixture_spec`: A mixture object with `levels` and `weights` properties
- `n_rows`: Number of rows to create

# Returns
Vector of mixture objects, all identical to the input specification

# Example
```julia
mixture = mix("A" => 0.3, "B" => 0.7)
col = create_mixture_column(mixture, 1000)  # 1000 rows of identical mixture
```

This is more efficient than `fill(mixture, n_rows)` for large datasets as it
avoids potential copying issues with complex mixture objects.
"""
function create_mixture_column(mixture_spec, n_rows::Int)
    if n_rows < 0
        throw(ArgumentError("Number of rows must be non-negative, got $n_rows"))
    end
    return fill(mixture_spec, n_rows)
end

"""
    expand_mixture_grid(base_data, mixture_specs::Dict{Symbol, Any})

Create all combinations of base data with mixture specifications for systematic 
marginal effects computation.

# Arguments
- `base_data`: Base data as NamedTuple or DataFrame-compatible structure
- `mixture_specs`: Dictionary mapping column names to mixture specifications

# Returns
Vector of NamedTuple data structures, each representing one combination

# Example
```julia
base_data = (x = [1.0, 2.0], y = [0.1, 0.2])
mixtures = Dict(
    :group => mix("A" => 0.5, "B" => 0.5),
    :treatment => mix("Control" => 0.3, "Treatment" => 0.7)
)

expanded = expand_mixture_grid(base_data, mixtures)
# Returns data with mixture columns added to each row
```

# Use Cases
- Reference grid creation for marginal effects
- Counterfactual analysis with multiple mixture variables
- Systematic sensitivity analysis across mixture specifications
"""
function expand_mixture_grid(base_data, mixture_specs::Dict{Symbol, <:Any})
    if isempty(mixture_specs)
        return [base_data]  # No mixtures to expand
    end
    
    # Ensure base_data is a NamedTuple
    if !(base_data isa NamedTuple)
        throw(ArgumentError("base_data must be a NamedTuple. Use Tables.columntable() to convert DataFrames."))
    end
    
    n_rows = length(first(values(base_data)))
    
    # Create expanded data with mixture columns
    expanded_data = Dict{Symbol, Any}()
    
    # Copy all base columns
    for (col_name, col_data) in pairs(base_data)
        expanded_data[col_name] = col_data
    end
    
    # Add mixture columns
    for (col_name, mixture_spec) in mixture_specs
        if haskey(expanded_data, col_name)
            @warn "Overriding existing column $col_name with mixture specification"
        end
        expanded_data[col_name] = create_mixture_column(mixture_spec, n_rows)
    end
    
    return [NamedTuple(expanded_data)]
end

"""
    validate_mixture_weights(weights::AbstractVector{<:Real}; atol::Real=1e-10)

Validate that mixture weights are properly normalized and non-negative.

# Arguments
- `weights`: Vector of mixture weights
- `atol`: Absolute tolerance for sum-to-one check

# Throws
- `ArgumentError`: If weights are invalid

# Example
```julia
validate_mixture_weights([0.3, 0.7])        # ✓ Valid
validate_mixture_weights([0.3, 0.6])        # ✗ Sum ≠ 1.0
validate_mixture_weights([0.5, -0.5])       # ✗ Negative weights
```
"""
function validate_mixture_weights(weights::AbstractVector{<:Real}; atol::Real=1e-10)
    if any(w < 0 for w in weights)
        throw(ArgumentError("Mixture weights must be non-negative: $weights"))
    end
    
    weight_sum = sum(weights)
    if !isapprox(weight_sum, 1.0, atol=atol)
        throw(ArgumentError("Mixture weights must sum to 1.0 (±$atol): got $weights (sum = $weight_sum)"))
    end
end

"""
    validate_mixture_levels(levels::AbstractVector)

Validate that mixture levels are unique and non-empty.

# Arguments
- `levels`: Vector of level identifiers

# Throws
- `ArgumentError`: If levels are invalid

# Example
```julia
validate_mixture_levels(["A", "B", "C"])    # ✓ Valid
validate_mixture_levels(["A", "A", "B"])    # ✗ Duplicate levels
validate_mixture_levels(String[])           # ✗ Empty levels
```
"""
function validate_mixture_levels(levels::AbstractVector)
    if isempty(levels)
        throw(ArgumentError("Mixture levels cannot be empty"))
    end
    
    if length(unique(levels)) != length(levels)
        duplicates = [level for level in unique(levels) if count(==(level), levels) > 1]
        throw(ArgumentError("Mixture levels must be unique. Duplicates found: $duplicates"))
    end
end

"""
    create_balanced_mixture(levels::AbstractVector)

Create a balanced (equal weight) mixture from a vector of levels.

# Arguments  
- `levels`: Vector of level identifiers

# Returns
Dictionary suitable for creating mixture objects: `Dict(level => weight, ...)`

# Example
```julia
balanced = create_balanced_mixture(["A", "B", "C"])
# Returns: Dict("A" => 0.333..., "B" => 0.333..., "C" => 0.333...)

# Use with mixture constructor:
mixture = mix(balanced...)  # Splat the dictionary
```

This is useful for creating reference mixtures where all levels should be
equally weighted for marginal effects computation.
"""
function create_balanced_mixture(levels::AbstractVector)
    if isempty(levels)
        throw(ArgumentError("Cannot create balanced mixture from empty levels"))
    end
    
    validate_mixture_levels(levels)
    
    n_levels = length(levels)
    weight = 1.0 / n_levels
    
    return Dict(string(level) => weight for level in levels)
end

"""
    _get_baseline_level(model, var) -> baseline_level

Extract baseline level from model's contrast coding (statistically principled).

This function implements the design decision to use the model's actual contrast
coding rather than making assumptions from the data.

# Arguments
- `model`: Fitted statistical model
- `var::Symbol`: Categorical variable name

# Returns
- Baseline level used in the model's contrast coding

# Throws
- `ArgumentError`: If baseline level cannot be determined

# Examples
```julia
baseline = _get_baseline_level(model, :region)  # Returns "North" if that's the baseline
```
"""
function _get_baseline_level(model, var::Symbol)
    # Use FormulaCompiler's proven approach: extract baseline from processed schema
    # Works for both GLM.jl and MixedModels.jl via unified StatsModels.jl pipeline
    
    model_formula = StatsModels.formula(model)
    
    # Handle different RHS structures
    matrix_term = if isa(model_formula.rhs, StatsModels.MatrixTerm)
        model_formula.rhs  # GLM case: RHS is MatrixTerm directly
    else
        model_formula.rhs[1]  # MixedModels case: RHS is tuple, MatrixTerm is first element
    end
    
    # Find CategoricalTerm for the requested variable
    for term in matrix_term.terms
        if isa(term, StatsModels.CategoricalTerm) && term.sym == var
            contrasts = term.contrasts
            baseline_levels = setdiff(contrasts.levels, StatsModels.coefnames(contrasts))
            if length(baseline_levels) == 1
                return baseline_levels[1]
            elseif length(baseline_levels) == 0
                throw(ArgumentError("No baseline level found for variable $var. All levels have coefficients."))
            else
                throw(ArgumentError("Multiple baseline levels found for variable $var: $baseline_levels"))
            end
        end
    end
    
    throw(ArgumentError("Could not find categorical variable $var in model formula terms. " *
                      "Ensure the variable is categorical and present in the model."))
end