# compile_term.jl - UPDATED FOR PHASE 2
#=
The functions in compile_term, turns each `AbstractTerm` in the model formula into a corresponding `AbstractEvaluator` so that the model matrix can be built efficiently.
- Allocates needed scratch space
- Builds position maps for terms
- UPDATED: Uses schema-based categorical contrasts from fitted model

See .md document.
=#

###############################################################################
# CATEGORICAL SCHEMA EXTRACTION SYSTEM
###############################################################################

"""
    CategoricalSchemaInfo

Complete categorical information extracted from fitted model schema.
Handles both main effects and interaction-only categorical variables.
"""
struct CategoricalSchemaInfo
    main_effect_contrasts::Union{Matrix{Float64}, Nothing}  # DummyCoding (k-1) - unchanged
    full_dummy_contrasts::Matrix{Float64}                   # NEW: FullDummyCoding (k) 
    n_levels::Int
    levels::Vector{String}
    level_codes::Vector{Int}
    column::Symbol
end

"""
    extract_complete_categorical_schema(model) -> Dict{Symbol, CategoricalSchemaInfo}

Properly extract categorical contrasts for both main effects and interactions.
The key insight is that we need to distinguish between:
1. Main effect contrasts (usually DummyCoding with k-1 columns)  
2. Interaction contrasts (may be FullDummyCoding with k columns for non-redundant vars)
"""
function extract_complete_categorical_schema(model)
    # # println("DEBUG: Schema extraction starting...")
    
    # Step 1: Get the actual fitted model matrix to understand the true structure
    fitted_matrix = StatsModels.modelmatrix(model)
    fitted_coefnames = StatsModels.coefnames(model)
    
    # # println("DEBUG: Fitted model has $(size(fitted_matrix, 2)) columns")
    # # println("DEBUG: Coefficient names: $(fitted_coefnames[1:min(10, end)])")
    
    # Step 2: Get the schema (this has the contrast matrices used during fitting)
    schema = get_model_schema(model)
    schema_dict = get_schema_dict(schema)
    
    categorical_info = Dict{Symbol, CategoricalSchemaInfo}()
    
    # Step 3: Extract each categorical variable's information
    for (term_key, term_info) in schema_dict
        if term_info isa CategoricalTerm
            col_symbol = extract_symbol_from_term(term_key, term_info)
            
            # println("DEBUG: Processing categorical variable: $col_symbol")
            
            # Get the contrast matrix that was ACTUALLY used during fitting
            fitted_contrast_matrix = Matrix{Float64}(term_info.contrasts.matrix)
            levels = collect(term_info.contrasts.levels)
            n_levels = length(levels)
            
            # println("DEBUG:   Levels: $levels")
            # println("DEBUG:   Fitted contrast matrix size: $(size(fitted_contrast_matrix))")
            
            # Step 4: Determine if this variable appears in main effects vs interactions
            main_effect_info = analyze_main_effect_usage(col_symbol, model, fitted_coefnames)
            interaction_info = analyze_interaction_usage(col_symbol, model, fitted_coefnames)
            
            # println("DEBUG:   Main effect usage: $(main_effect_info.has_main_effect)")
            # println("DEBUG:   Interaction usage: $(interaction_info.appears_in_interactions)")
            
            # Step 5: Determine correct contrast matrices
            main_contrasts, full_dummy_contrasts = determine_contrast_matrices(
                col_symbol, 
                fitted_contrast_matrix, 
                levels,
                main_effect_info,
                interaction_info
            )

            # println("DEBUG:   Main contrasts: $(main_contrasts === nothing ? "none" : size(main_contrasts))")
            # println("DEBUG:   Full dummy contrasts: $(size(full_dummy_contrasts))")

            # Step 6: Store both contrast types
            categorical_info[col_symbol] = CategoricalSchemaInfo(
                main_contrasts,           # May be nothing for interaction-only variables
                full_dummy_contrasts,     # Always present - FullDummyCoding
                n_levels,
                levels,
                Int[],                    # Will populate with data later
                col_symbol
            )
        end
    end
    
    # println("DEBUG: Schema extraction complete: $(length(categorical_info)) variables")
    return categorical_info
end

"""
    analyze_main_effect_usage(col_symbol::Symbol, model, coefnames::Vector) -> NamedTuple

Analyze whether a categorical variable appears as a main effect in the fitted model.
"""
function analyze_main_effect_usage(col_symbol::Symbol, model, coefnames::Vector)
    # Look for coefficient names that correspond to main effects of this variable
    # Main effect names typically look like "variable: level" 
    main_effect_pattern = Regex("^$(col_symbol):")
    
    main_effect_coefs = filter(name -> occursin(main_effect_pattern, name), coefnames)
    has_main_effect = !isempty(main_effect_coefs)
    
    return (
        has_main_effect = has_main_effect,
        main_effect_coefs = main_effect_coefs,
        n_main_contrasts = length(main_effect_coefs)
    )
end

"""
    analyze_interaction_usage(col_symbol::Symbol, model, coefnames::Vector) -> NamedTuple

Analyze whether a categorical variable appears in interactions in the fitted model.
"""
function analyze_interaction_usage(col_symbol::Symbol, model, coefnames::Vector)
    # Look for coefficient names that contain this variable in interactions
    # Interaction names typically look like "var1: level1 & var2: level2"
    interaction_pattern = Regex("$(col_symbol):")
    ampersand_pattern = r"&"
    
    interaction_coefs = filter(coefnames) do name
        # Must contain the variable name AND an ampersand (indicating interaction)
        occursin(interaction_pattern, name) && occursin(ampersand_pattern, name)
    end
    
    appears_in_interactions = !isempty(interaction_coefs)
    
    return (
        appears_in_interactions = appears_in_interactions,
        interaction_coefs = interaction_coefs,
        n_interaction_terms = length(interaction_coefs)
    )
end

"""
    determine_contrast_matrices(
        col_symbol::Symbol,
        fitted_matrix::Matrix{Float64},
        levels::Vector,
        main_info::NamedTuple,
        interaction_info::NamedTuple
    ) -> Tuple{Union{Matrix{Float64}, Nothing}, Matrix{Float64}}

Determine the correct contrast matrices for main effects and interactions.
"""
function determine_contrast_matrices(
    col_symbol::Symbol,
    fitted_matrix::Matrix{Float64},
    levels::Vector,
    main_info::NamedTuple,
    interaction_info::NamedTuple
)
    n_levels = length(levels)
    
    # Always create FullDummyCoding for potential interaction use
    full_dummy_contrasts = create_full_dummy_matrix(levels)

    if main_info.has_main_effect
        # Variable has main effect - store the fitted contrasts (DummyCoding)
        main_contrasts = fitted_matrix
        # println("DEBUG:     Variable has main effect - storing DummyCoding: $(size(fitted_matrix))")
    else
        # Variable is interaction-only - no main effect contrasts
        main_contrasts = nothing
        # println("DEBUG:     Variable is interaction-only - no main effect contrasts")
    end

    # println("DEBUG:     Created FullDummyCoding: $(size(full_dummy_contrasts))")
    return main_contrasts, full_dummy_contrasts
end

"""
    extract_symbol_from_term(term_key, term_info) -> Symbol

Extract column symbol from various term key types.
"""
function extract_symbol_from_term(term_key, term_info)
    if term_key isa Symbol
        return term_key
    elseif term_key isa Term
        return term_key.sym
    elseif hasfield(typeof(term_info), :sym)
        return term_info.sym
    else
        error("Cannot extract column symbol from term_key: $term_key ($(typeof(term_key)))")
    end
end

"""
    extract_interaction_contrasts_from_formula!(categorical_info::Dict{Symbol, CategoricalSchemaInfo}, model)

Extract interaction-specific contrast matrices from the model's formula.
FIXED: Properly extracts FullDummyCoding for non-redundant variables.
"""
function extract_interaction_contrasts_from_formula!(categorical_info::Dict{Symbol, CategoricalSchemaInfo}, model)
    # println("DEBUG: Extracting interaction contrasts from model formula")
    
    # Get the fitted model's formula
    formula = StatsModels.formula(model)
    # println("DEBUG: Model formula: $formula")
    
    # Get the model's schema to see what contrasts were actually used
    schema = get_model_schema(model)
    schema_dict = get_schema_dict(schema)
    
    # Extract interaction terms from the formula
    interaction_terms = extract_interaction_terms(formula.rhs)
    # println("DEBUG: Found $(length(interaction_terms)) interaction terms")
    
    for (i, interaction_term) in enumerate(interaction_terms)
        # println("DEBUG: Processing interaction term $i: $(typeof(interaction_term))")
        
        # Check which categorical variables appear in this interaction
        categorical_in_interaction = Symbol[]
        for component in interaction_term.terms
            if component isa CategoricalTerm
                push!(categorical_in_interaction, component.sym)
            end
        end
        
        # println("DEBUG: Categoricals in this interaction: $categorical_in_interaction")
        
        # For each categorical in the interaction, check if it needs FullDummyCoding
        for component in interaction_term.terms
            if component isa CategoricalTerm
                col_symbol = component.sym
                
                # Check if this variable has a main effect in the formula
                has_main_effect = check_has_main_effect(formula.rhs, col_symbol)
                
                # println("DEBUG: Variable $col_symbol:")
                println("  Has main effect: $has_main_effect")
                
                # The key insight: if no main effect, it uses FullDummyCoding in interactions
                if !has_main_effect
                    # println("DEBUG: $col_symbol is NON-REDUNDANT in interaction (no main effect)")
                    
                    # This variable should use FullDummyCoding in interactions
                    if haskey(categorical_info, col_symbol)
                        old_info = categorical_info[col_symbol]
                        
                        # Create FullDummyCoding contrast matrix
                        n_levels = old_info.n_levels
                        full_contrast_matrix = Matrix{Float64}(I, n_levels, n_levels)
                        
                        # println("DEBUG: Creating FullDummyCoding for $col_symbol: $(size(full_contrast_matrix))")
                        
                        updated_info = CategoricalSchemaInfo(
                            nothing,                    # No main effect contrasts
                            full_contrast_matrix,       # FullDummyCoding for interactions
                            old_info.n_levels,
                            old_info.levels,
                            old_info.level_codes,
                            old_info.column
                        )
                        
                        categorical_info[col_symbol] = updated_info
                    else
                        # New interaction-only variable
                        # Extract from the actual component contrasts
                        interaction_contrast_matrix = Matrix{Float64}(component.contrasts.matrix)
                        levels = collect(component.contrasts.levels)
                        n_levels = length(levels)
                        
                        # println("DEBUG: New interaction-only variable: $col_symbol")
                        # println("DEBUG: Contrast matrix from model: $(size(interaction_contrast_matrix))")
                        
                        categorical_info[col_symbol] = CategoricalSchemaInfo(
                            nothing,                        # No main effect
                            interaction_contrast_matrix,   # Use what the model actually used
                            n_levels,
                            levels,
                            Int[],                         # Will populate with actual data
                            col_symbol
                        )
                    end
                else
                    # Has main effect - keep using DummyCoding for interactions too
                    # println("DEBUG: $col_symbol has main effect, keeping DummyCoding")
                end
            end
        end
    end
    
    return nothing
end

"""
    check_has_main_effect(rhs_term, col_symbol::Symbol) -> Bool

Check if a categorical variable appears as a main effect in the formula.
"""
function check_has_main_effect(rhs_term, col_symbol::Symbol)
    if rhs_term isa CategoricalTerm && rhs_term.sym == col_symbol
        return true
    elseif rhs_term isa MatrixTerm
        for sub_term in rhs_term.terms
            # Only check non-interaction terms
            if !(sub_term isa InteractionTerm) && check_has_main_effect(sub_term, col_symbol)
                return true
            end
        end
    end
    return false
end

"""
    extract_interaction_terms(rhs_term) -> Vector{InteractionTerm}

Recursively extract all InteractionTerm objects from the formula RHS.
"""
function extract_interaction_terms(rhs_term)
    interaction_terms = InteractionTerm[]
    extract_interaction_terms_recursive!(interaction_terms, rhs_term)
    return interaction_terms
end

"""
    extract_interaction_terms_recursive!(result::Vector{InteractionTerm}, term)

Recursively collect InteractionTerm objects.
"""
function extract_interaction_terms_recursive!(result::Vector{InteractionTerm}, term::InteractionTerm)
    push!(result, term)
end

function extract_interaction_terms_recursive!(result::Vector{InteractionTerm}, term::MatrixTerm)
    for sub_term in term.terms
        extract_interaction_terms_recursive!(result, sub_term)
    end
end

function extract_interaction_terms_recursive!(result::Vector{InteractionTerm}, term::Union{CategoricalTerm, ContinuousTerm, Term, InterceptTerm, ConstantTerm, FunctionTerm, ZScoredTerm})
    # Not an interaction term - skip
end

"""
    get_model_schema(model)

Get the schema from different model types (LinearModel, GLM, MixedModel, etc.)
"""
function get_model_schema(model::StatsModels.TableRegressionModel)
    # For TableRegressionModel (includes LinearModel, GeneralizedLinearModel)
    return model.mf.schema
end

function get_model_schema(model::Union{LinearMixedModel, GeneralizedLinearMixedModel})
    # For MixedModels
    return model.formula.schema  # or wherever MixedModels stores schema
end

function get_model_schema(model)
    # Generic fallback - try common locations
    if hasfield(typeof(model), :mf) && hasfield(typeof(model.mf), :schema)
        return model.mf.schema
    elseif hasfield(typeof(model), :schema)
        return model.schema
    elseif hasfield(typeof(model), :model) && hasfield(typeof(model.model), :schema)
        return model.model.schema
    else
        # Debug what fields are available
        # # println("DEBUG: Model fields: $(fieldnames(typeof(model)))")
        if hasfield(typeof(model), :mf)
            # # println("DEBUG: model.mf fields: $(fieldnames(typeof(model.mf)))")
        end
        error("Cannot locate schema in model of type $(typeof(model)). Available fields: $(fieldnames(typeof(model)))")
    end
end

"""
    get_schema_dict(schema)

Get the dictionary mapping column names to term info from schema.
"""
function get_schema_dict(schema)
    if hasfield(typeof(schema), :schema)
        return schema.schema
    elseif hasfield(typeof(schema), :terms)
        return schema.terms
    else
        # # println("DEBUG: Schema fields: $(fieldnames(typeof(schema)))")
        error("Cannot locate schema dictionary in schema of type $(typeof(schema)). Available fields: $(fieldnames(typeof(schema)))")
    end
end

"""
    populate_level_codes_from_data!(
        categorical_schema::Dict{Symbol, CategoricalSchemaInfo}, 
        data::NamedTuple
    )

Populate the level_codes field in each CategoricalSchemaInfo with actual data.
This replaces the old prepare_categorical_levels function.
"""
function populate_level_codes_from_data!(
    categorical_schema::Dict{Symbol, CategoricalSchemaInfo}, 
    data::NamedTuple
)
    # # println("DEBUG: Populating level codes from data")
    
    for (col_name, schema_info) in categorical_schema
        if haskey(data, col_name)
            col_data = data[col_name]
            
            if col_data isa CategoricalVector
                # Pre-extract all level codes - allocate once during compilation
                level_codes = [levelcode(val) for val in col_data]
                
                # Update the schema info with actual level codes
                updated_info = CategoricalSchemaInfo(
                    schema_info.main_effect_contrasts,
                    schema_info.full_dummy_contrasts,  # ← FIXED: Use new field name
                    schema_info.n_levels,
                    schema_info.levels,
                    level_codes,  # Now populated from data
                    schema_info.column
                )

                categorical_schema[col_name] = updated_info
                
                # # println("DEBUG:   Column $col_name: extracted $(length(level_codes)) level codes")
            else
                @warn "Column $col_name is not categorical in data but is categorical in model schema"
            end
        else
            @warn "Column $col_name found in model schema but not in data"
        end
    end
    
    return nothing
end

"""
    determine_main_effect_contrasts!(categorical_schema::Dict{Symbol, CategoricalSchemaInfo}, model)

Determine which categorical variables have main effects vs interaction-only.
Updates the main_effect_contrasts field appropriately.
"""
function determine_main_effect_contrasts!(
    categorical_schema::Dict{Symbol, CategoricalSchemaInfo}, 
    model
)
    # # println("DEBUG: Determining main effect vs interaction-only contrasts")
    
    # Get the fixed effects formula (strips random effects for mixed models)
    fixed_formula = fixed_effects_form(model)
    main_effect_terms = extract_main_effect_terms(fixed_formula.rhs)
    
    # # println("DEBUG: Main effect categorical terms: $main_effect_terms")
    
    for (col_name, schema_info) in categorical_schema
        has_main_effect = col_name in main_effect_terms
        
        if has_main_effect
            # This variable has a main effect - use the schema contrasts for main effects
            main_contrasts = schema_info.main_effect_contrasts  # Keep existing
            # # println("DEBUG:   $col_name: HAS main effect")
        else
            # This variable is interaction-only - set main_effect_contrasts to nothing
            updated_info = CategoricalSchemaInfo(
                nothing,                        # No main effect contrasts
                schema_info.full_dummy_contrasts,
                schema_info.n_levels,
                schema_info.levels,
                schema_info.level_codes,
                schema_info.column
            )
            
            categorical_schema[col_name] = updated_info
            # # println("DEBUG:   $col_name: interaction-only (no main effect)")
        end
    end
    
    return nothing
end

"""
    extract_main_effect_terms(rhs_term) -> Vector{Symbol}

Extract categorical column names that appear as main effects (not just interactions).
"""
function extract_main_effect_terms(rhs_term)
    main_effect_categoricals = Symbol[]
    extract_main_effect_terms_recursive!(main_effect_categoricals, rhs_term)
    return unique(main_effect_categoricals)
end

"""
    extract_main_effect_terms_recursive!(result::Vector{Symbol}, term)

Recursively extract main effect categorical terms.
"""
function extract_main_effect_terms_recursive!(result::Vector{Symbol}, term::CategoricalTerm)
    push!(result, term.sym)
end

function extract_main_effect_terms_recursive!(result::Vector{Symbol}, term::Union{ContinuousTerm, Term})
    # Not categorical - skip
end

function extract_main_effect_terms_recursive!(result::Vector{Symbol}, term::InteractionTerm)
    # Don't extract from interactions - these are not main effects
end

function extract_main_effect_terms_recursive!(result::Vector{Symbol}, term::MatrixTerm)
    for sub_term in term.terms
        extract_main_effect_terms_recursive!(result, sub_term)
    end
end

function extract_main_effect_terms_recursive!(result::Vector{Symbol}, term::FunctionTerm)
    for arg in term.args
        extract_main_effect_terms_recursive!(result, arg)
    end
end

function extract_main_effect_terms_recursive!(result::Vector{Symbol}, term::Union{InterceptTerm, ConstantTerm})
    # No categorical columns
end

function extract_main_effect_terms_recursive!(result::Vector{Symbol}, term::ZScoredTerm)
    extract_main_effect_terms_recursive!(result, term.term)
end

"""
    validate_categorical_schema(categorical_schema::Dict{Symbol, CategoricalSchemaInfo})

Validate that the extracted schema information is consistent and complete.
"""
function validate_categorical_schema(categorical_schema::Dict{Symbol, CategoricalSchemaInfo})
    # # println("DEBUG: Validating categorical schema")
    
    for (col_name, schema_info) in categorical_schema
        # # println("DEBUG: Validating column: $col_name")
        
        # Check basic consistency
        if schema_info.n_levels != length(schema_info.levels)
            error("Schema inconsistency for $col_name: n_levels=$(schema_info.n_levels) but levels=$(length(schema_info.levels))")
        end
        
        if !isempty(schema_info.level_codes)
            if length(schema_info.level_codes) == 0
                @warn "No level codes extracted for $col_name"
            else
                # # println("DEBUG:   Level codes: $(length(schema_info.level_codes)) values")
            end
        end
        
        # Check contrast matrices
        full_dummy_size = size(schema_info.full_dummy_contrasts)
        expected_rows = schema_info.n_levels

        if full_dummy_size[1] != expected_rows
            error("Full dummy contrast matrix for $col_name has $(full_dummy_size[1]) rows, expected $expected_rows")
        end

        # # println("DEBUG:   Full dummy contrasts: $(full_dummy_size)")
        
        if schema_info.main_effect_contrasts !== nothing
            main_size = size(schema_info.main_effect_contrasts)
            # # println("DEBUG:   Main effect contrasts: $(main_size)")
            
            if main_size[1] != expected_rows
                error("Main effect contrast matrix for $col_name has $(main_size[1]) rows, expected $expected_rows")
            end
        else
            # # println("DEBUG:   No main effect contrasts (interaction-only)")
        end
    end
    
    # # println("DEBUG: Schema validation complete")
    return true
end

"""
    create_full_dummy_matrix(levels::Vector) -> Matrix{Float64}

Create a full dummy coding matrix (k×k identity matrix for k levels).
This is what StatsModels calls FullDummyCoding.
"""
function create_full_dummy_matrix(levels::Vector)
    n_levels = length(levels)
    return Matrix{Float64}(I, n_levels, n_levels)  # Identity matrix
end

###############################################################################
# MAIN COMPILATION FUNCTION
###############################################################################

"""
    compile_term(
        term::AbstractTerm, start_position::Int = 1, 
        scratch_allocator::ScratchAllocator = ScratchAllocator(),
        categorical_schema::Dict{Symbol, CategoricalSchemaInfo} = Dict{Symbol, CategoricalSchemaInfo}()
    ) -> AbstractEvaluator

UPDATED: Compile term into self-contained evaluator using schema-based categorical contrasts.
Now uses the fitted model's exact contrast matrices instead of generating new ones.
"""
function compile_term(
    term::AbstractTerm, 
    start_position::Int = 1, 
    scratch_allocator::ScratchAllocator = ScratchAllocator(),
    categorical_schema::Dict{Symbol, CategoricalSchemaInfo} = Dict{Symbol, CategoricalSchemaInfo}()
)

    if term isa InterceptTerm
        evaluator = ConstantEvaluator(hasintercept(term) ? 1.0 : 0.0, start_position)
        return evaluator

    elseif term isa ConstantTerm
        evaluator = ConstantEvaluator(Float64(term.n), start_position)
        return evaluator
        
    elseif term isa Union{ContinuousTerm, Term}
        evaluator = ContinuousEvaluator(term.sym, start_position)
        return evaluator
        
    elseif term isa CategoricalTerm
        # # println("DEBUG: Compiling CategoricalTerm for $(term.sym)")
        
        # UPDATED: Use schema-extracted contrasts instead of generating new ones
        if haskey(categorical_schema, term.sym)
            schema_info = categorical_schema[term.sym]
            
            # Use main effect contrasts (DummyCoding) for main effect terms
            contrast_matrix = schema_info.main_effect_contrasts
            level_codes = schema_info.level_codes
            
            if contrast_matrix === nothing
                error("No main effect contrasts available for $(term.sym) (interaction-only variable used as main effect)")
            end
            
            # # println("DEBUG: Using schema contrasts for $(term.sym)")
            # # println("DEBUG: Contrast matrix size: $(size(contrast_matrix))")
            
        else
            error("Categorical variable $(term.sym) not found in schema. Available: $(keys(categorical_schema))")
        end
        
        n_contrasts = size(contrast_matrix, 2)
        positions = collect(start_position:(start_position + n_contrasts - 1))
        
        evaluator = CategoricalEvaluator(
            term.sym,
            contrast_matrix,        # FROM SCHEMA - no more independent generation!
            schema_info.n_levels,
            positions,
            level_codes             # FROM SCHEMA - pre-extracted level codes
        )
        
        # # println("DEBUG: Created CategoricalEvaluator with $(n_contrasts) contrasts")
        return evaluator
        
    elseif term isa FunctionTerm
        arg_evaluators = AbstractEvaluator[]
        arg_scratch_map = UnitRange{Int}[]
        
        for arg in term.args
            arg_width = 1 # Functions always produce scalar output
            arg_scratch = allocate_scratch!(scratch_allocator, arg_width)
            arg_start_pos = first(arg_scratch)
            
            # Pass categorical_schema through recursive call
            arg_eval = compile_term(arg, arg_start_pos, scratch_allocator, categorical_schema)
            
            push!(arg_evaluators, arg_eval)
            push!(arg_scratch_map, arg_scratch)
        end
        
        all_scratch = Int[]
        for range in arg_scratch_map
            append!(all_scratch, collect(range))
        end
        
        return FunctionEvaluator(
            term.f,
            arg_evaluators,
            start_position,
            all_scratch,
            arg_scratch_map
        )
        
    elseif term isa InteractionTerm
        # # println("DEBUG: Compiling InteractionTerm with $(length(term.terms)) components")
        
        component_evaluators = AbstractEvaluator[]
        component_widths = Int[]
        
        # CRITICAL FIX: Ensure each component uses the correct contrast matrix
        for (i, comp) in enumerate(term.terms)
            # # println("DEBUG: Processing interaction component $i: $(typeof(comp))")
            
            if comp isa CategoricalTerm
                # # println("DEBUG: Categorical component: $(comp.sym)")
                
                if haskey(categorical_schema, comp.sym)
                    schema_info = categorical_schema[comp.sym]
                    
                    # CRITICAL: Use INTERACTION contrasts for interaction components
                    interaction_contrast_matrix = schema_info.interaction_contrasts
                    level_codes = schema_info.level_codes
                    
                    # # println("DEBUG: Using INTERACTION contrasts for $(comp.sym)")
                    # # println("DEBUG: Interaction contrast matrix size: $(size(interaction_contrast_matrix))")
                    # # println("DEBUG: First few rows of contrast matrix:")
                    display(interaction_contrast_matrix[1:min(3, end), :])
                    
                    n_interaction_contrasts = size(interaction_contrast_matrix, 2)
                    
                    # IMPORTANT: These positions are temporary and will be overridden
                    # The actual positions come from the InteractionEvaluator
                    temp_positions = collect(1:n_interaction_contrasts)
                    
                    interaction_categorical_eval = CategoricalEvaluator(
                        comp.sym,
                        interaction_contrast_matrix,    # INTERACTION contrasts (may be FullDummyCoding)
                        schema_info.n_levels,
                        temp_positions,                 # Temporary positions
                        level_codes                      # Pre-extracted level codes
                    )
                    
                    push!(component_evaluators, interaction_categorical_eval)
                    push!(component_widths, n_interaction_contrasts)
                    
                    # # println("DEBUG: Created interaction categorical component:")
                    # println("  Column: $(comp.sym)")
                    # println("  Levels: $(schema_info.n_levels)")
                    # println("  Contrasts: $(n_interaction_contrasts)")
                    # println("  Contrast type: $(n_interaction_contrasts == schema_info.n_levels ? "FullDummyCoding" : "DummyCoding")")
                    
                else
                    error("Categorical variable $(comp.sym) not found in schema for interaction")
                end
                
            else
                # Handle non-categorical components (continuous, functions, etc.)
                temp_allocator = ScratchAllocator()
                comp_eval = compile_term(comp, 1, temp_allocator, categorical_schema)
                
                push!(component_evaluators, comp_eval)
                push!(component_widths, output_width(comp_eval))
                
                # # println("DEBUG: Created non-categorical component with width $(output_width(comp_eval))")
            end
        end
        
        # Convert to compile-time tuples
        N = length(component_evaluators)
        components_tuple = ntuple(i -> component_evaluators[i], N)
        widths_tuple = ntuple(i -> component_widths[i], N)
        
        # Calculate output setup
        total_width = prod(component_widths)
        positions = collect(start_position:(start_position + total_width - 1))
        
        # # println("DEBUG: Interaction compilation summary:")
        # println("  Total components: $N")
        # println("  Component widths: $component_widths")
        # println("  Total width: $total_width (= $(join(component_widths, " × ")))")
        # println("  Output positions: $(length(positions)) positions starting at $start_position")
        
        # Create fully typed InteractionEvaluator
        return InteractionEvaluator{N, typeof(components_tuple), typeof(widths_tuple)}(
            components_tuple,      # NTuple{N, AbstractEvaluator}
            widths_tuple,          # NTuple{N, Int}
            positions,
            start_position,
            total_width
        )
    elseif term isa ZScoredTerm
        underlying_width = width(term.term)
        underlying_scratch = allocate_scratch!(scratch_allocator, underlying_width)
        underlying_start_pos = first(underlying_scratch)
        
        # Pass categorical_schema through recursive call
        underlying_eval = compile_term(term.term, underlying_start_pos, scratch_allocator, categorical_schema)
        
        positions = collect(start_position:(start_position + underlying_width - 1))
        center = term.center isa Number ? Float64(term.center) : Float64(term.center[1])
        scale = term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
        
        return ZScoreEvaluator(
            underlying_eval,
            center,
            scale,
            positions,
            collect(underlying_scratch),
            underlying_scratch
        )
        
    elseif term isa MatrixTerm
        # # println("DEBUG: Compiling MatrixTerm with $(length(term.terms)) sub-terms")
        
        # PHASE 1: Collect main effect categoricals for context
        main_effect_vars = collect_main_effect_categoricals(term)
        
        sub_evaluators = AbstractEvaluator[]
        current_pos = start_position
        max_scratch = 0
        
        # PHASE 2: Compile sub-terms with context awareness
        for sub_term in term.terms
            if width(sub_term) > 0
                if sub_term isa InteractionTerm
                    # Use context-aware compilation for interactions
                    sub_eval = compile_interaction_term_with_context(
                        sub_term, current_pos, scratch_allocator, 
                        categorical_schema, main_effect_vars
                    )
                else
                    # Regular compilation for non-interactions  
                    sub_eval = compile_term(sub_term, current_pos, scratch_allocator, categorical_schema)
                end
                
                next_pos = current_pos + output_width(sub_eval)
                push!(sub_evaluators, sub_eval)
                current_pos = next_pos
                
                sub_scratch = max_scratch_needed(sub_eval)
                max_scratch = max(max_scratch, sub_scratch)
            end
        end
        
        # PHASE 3: Categorize evaluators and create CombinedEvaluator
        total_width = current_pos - start_position
        
        constant_ops = PrecomputedConstantOp[]
        continuous_ops = PrecomputedContinuousOp[]
        categorical_evals = CategoricalEvaluator[]
        function_evals = FunctionEvaluator[]
        interaction_evals = InteractionEvaluator[]
        
        for eval in sub_evaluators
            if eval isa ConstantEvaluator
                push!(constant_ops, PrecomputedConstantOp(eval.value, eval.position))
            elseif eval isa ContinuousEvaluator
                push!(continuous_ops, PrecomputedContinuousOp(eval.column, eval.position))
            elseif eval isa CategoricalEvaluator
                push!(categorical_evals, eval)
            elseif eval isa FunctionEvaluator
                push!(function_evals, eval)
            elseif eval isa InteractionEvaluator
                push!(interaction_evals, eval)
            else
                error("Unknown evaluator type in matrix term: $(typeof(eval))")
            end
        end
        
        # # println("DEBUG: MatrixTerm compiled: $(length(constant_ops)) constants, $(length(continuous_ops)) continuous, $(length(categorical_evals)) categorical, $(length(function_evals)) functions, $(length(interaction_evals)) interactions")
        
        return CombinedEvaluator(
            constant_ops,
            continuous_ops,
            categorical_evals,
            function_evals,
            interaction_evals,
            total_width,
            max_scratch
        )
    else
        error("Unknown term type: $(typeof(term))")
    end
end


"""
    collect_main_effect_categoricals(matrix_term::MatrixTerm) -> Set{Symbol}

Collect categorical variables that appear as main effects in this MatrixTerm.
"""
function collect_main_effect_categoricals(matrix_term::MatrixTerm)
    main_effects = Set{Symbol}()
    
    for sub_term in matrix_term.terms
        if sub_term isa CategoricalTerm
            push!(main_effects, sub_term.sym)
            # # println("DEBUG: Found main effect: $(sub_term.sym)")
        elseif sub_term isa Union{ContinuousTerm, Term}
            # These don't affect categorical contrast choices
        elseif sub_term isa InteractionTerm
            # Skip - we're only collecting main effects
        elseif sub_term isa InterceptTerm || sub_term isa ConstantTerm
            # Skip - not relevant
        else
            # # println("DEBUG: Unknown sub_term type in main effect collection: $(typeof(sub_term))")
        end
    end
    
    # # println("DEBUG: Collected main effect categoricals: $main_effects")
    return main_effects
end

"""
    compile_interaction_term_with_context(
        term::InteractionTerm,
        start_position::Int,
        scratch_allocator::ScratchAllocator,
        categorical_schema::Dict{Symbol, CategoricalSchemaInfo},
        main_effect_vars::Set{Symbol}
    ) -> InteractionEvaluator

Compile InteractionTerm with knowledge of which variables have main effects.
"""
function compile_interaction_term_with_context(
    term::InteractionTerm,
    start_position::Int,
    scratch_allocator::ScratchAllocator,
    categorical_schema::Dict{Symbol, CategoricalSchemaInfo},
    main_effect_vars::Set{Symbol}
)
    # # println("DEBUG: Compiling interaction with context: main effects = $main_effect_vars")
    
    component_evaluators = AbstractEvaluator[]
    component_widths = Int[]
    
    # Process each interaction component with context-aware contrast selection
    for (i, comp) in enumerate(term.terms)
        # # println("DEBUG: Processing interaction component $i: $(typeof(comp))")
        
        if comp isa CategoricalTerm
            # # println("DEBUG: Categorical component: $(comp.sym)")
            
            if haskey(categorical_schema, comp.sym)
                schema_info = categorical_schema[comp.sym]
                
                # CRITICAL: Choose contrasts based on main effect presence
                if comp.sym in main_effect_vars
                    # Variable has main effect - use same contrasts as main effect
                    contrast_matrix = schema_info.main_effect_contrasts
                    # # println("DEBUG: $(comp.sym) HAS main effect - using DummyCoding $(size(contrast_matrix))")
                else
                    # Variable has no main effect - use FullDummyCoding
                    contrast_matrix = schema_info.full_dummy_contrasts
                    # # println("DEBUG: $(comp.sym) NO main effect - using FullDummyCoding $(size(contrast_matrix))")
                end
                
                level_codes = schema_info.level_codes
                n_contrasts = size(contrast_matrix, 2)
                temp_positions = collect(1:n_contrasts)  # Temp positions
                
                interaction_categorical_eval = CategoricalEvaluator(
                    comp.sym,
                    contrast_matrix,    # Context-appropriate contrasts
                    schema_info.n_levels,
                    temp_positions,
                    level_codes
                )
                
                push!(component_evaluators, interaction_categorical_eval)
                push!(component_widths, n_contrasts)
                
                # # println("DEBUG: Created interaction categorical component with $(n_contrasts) contrasts")
                
            else
                error("Categorical variable $(comp.sym) not found in schema for interaction")
            end
            
        else
            # Handle non-categorical components (continuous, functions, etc.)
            temp_allocator = ScratchAllocator()
            comp_eval = compile_term(comp, 1, temp_allocator, categorical_schema)
            
            push!(component_evaluators, comp_eval)
            push!(component_widths, output_width(comp_eval))
            
            # # println("DEBUG: Created non-categorical component with width $(output_width(comp_eval))")
        end
    end
    
    # Rest is identical to existing InteractionTerm compilation
    N = length(component_evaluators)
    components_tuple = ntuple(i -> component_evaluators[i], N)
    widths_tuple = ntuple(i -> component_widths[i], N)
    
    total_width = prod(component_widths)
    positions = collect(start_position:(start_position + total_width - 1))
    
    # # println("DEBUG: Context-aware interaction total width: $total_width ($(join(component_widths, " × ")))")
    
    return InteractionEvaluator{N, typeof(components_tuple), typeof(widths_tuple)}(
        components_tuple,
        widths_tuple,
        positions,
        start_position,
        total_width
    )
end

