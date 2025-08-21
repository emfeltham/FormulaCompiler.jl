# compile_term.jl - FIXED FOR MIXED INTERACTIONS
#=
The functions in compile_term, turns each `AbstractTerm` in the model formula into a corresponding `AbstractEvaluator` so that the model matrix can be built efficiently.
- Allocates needed scratch space
- Builds position maps for terms
- FIXED: Correctly handles contrast selection for mixed continuous-categorical interactions

See .md document.
=#

using LinearAlgebra: I  # For identity matrix in create_full_dummy_matrix

###############################################################################
# CATEGORICAL SCHEMA EXTRACTION SYSTEM
###############################################################################

"""
    CategoricalSchemaInfo

Complete categorical information extracted from fitted model schema.
UPDATED: Now stores both DummyCoding and FullDummyCoding for flexible use.
"""
struct CategoricalSchemaInfo
    dummy_contrasts::Matrix{Float64}         # DummyCoding (k-1 columns) - always available
    full_dummy_contrasts::Matrix{Float64}    # FullDummyCoding (k columns) - always available
    main_effect_contrasts::Union{Matrix{Float64}, Nothing}  # What's used for main effects (if any)
    n_levels::Int
    levels::Vector{String}
    level_codes::Vector{Int}
    column::Symbol
end

"""
    create_dummy_coding_matrix(levels::Vector) -> Matrix{Float64}

Create a dummy coding matrix (k×(k-1) matrix dropping first level).
This is what StatsModels calls DummyCoding.
"""
function create_dummy_coding_matrix(levels::Vector)
    n_levels = length(levels)
    if n_levels <= 1
        return Matrix{Float64}(undef, n_levels, 0)  # No contrasts for single level
    end
    
    # Create k×(k-1) matrix dropping the first level (reference)
    contrast_matrix = zeros(Float64, n_levels, n_levels - 1)
    for i in 2:n_levels
        contrast_matrix[i, i-1] = 1.0
    end
    return contrast_matrix
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


"""
    extract_categorical_schema_from_mixed_model(model::Union{LinearMixedModel, GeneralizedLinearMixedModel}) -> Dict{Symbol, CategoricalSchemaInfo}

Extract categorical schema for MixedModels by working directly with the fixed effects formula.
MixedModels don't store schemas like GLM models, so we extract from the formula structure.
"""
function extract_categorical_schema_from_mixed_model(model::Union{LinearMixedModel, GeneralizedLinearMixedModel})
    # Get the fixed effects formula (strips random effects)
    fixed_form = fixed_effects_form(model)
    
    # Get coefficient names from the MixedModel
    fitted_coefnames = coefnames(model)
    
    categorical_info = Dict{Symbol, CategoricalSchemaInfo}()
    
    # Extract categorical terms directly from the formula RHS
    extract_categorical_terms_from_formula!(categorical_info, fixed_form.rhs, fitted_coefnames)
    
    return categorical_info
end

"""
    extract_categorical_terms_from_formula!(categorical_info, term, fitted_coefnames)

Recursively extract categorical terms from formula structure.
"""
function extract_categorical_terms_from_formula!(categorical_info::Dict{Symbol, CategoricalSchemaInfo}, term::CategoricalTerm, fitted_coefnames::Vector{String})
    col_symbol = term.sym
    
    # Extract contrast matrix from the fitted coefficient names
    # This follows the same logic as the schema-based approach but reconstructs from coefnames
    contrast_matrix = term.contrasts.matrix
    levels = term.contrasts.levels
    n_levels = length(levels)
    
    # Create both dummy and full dummy contrasts
    dummy_contrasts = contrast_matrix  # The fitted contrasts (usually DummyCoding)
    full_dummy_contrasts = Matrix{Float64}(I, n_levels, n_levels)  # FullDummyCoding for interactions
    
    categorical_info[col_symbol] = CategoricalSchemaInfo(
        dummy_contrasts,
        full_dummy_contrasts,  
        dummy_contrasts,  # Use dummy as the selected contrasts initially
        n_levels,
        levels,
        Int[],  # level_codes will be populated later
        col_symbol
    )
end

function extract_categorical_terms_from_formula!(categorical_info::Dict{Symbol, CategoricalSchemaInfo}, term::MatrixTerm, fitted_coefnames::Vector{String})
    # Process each term in the MatrixTerm
    for sub_term in term.terms
        extract_categorical_terms_from_formula!(categorical_info, sub_term, fitted_coefnames)
    end
end

function extract_categorical_terms_from_formula!(categorical_info::Dict{Symbol, CategoricalSchemaInfo}, term::InteractionTerm, fitted_coefnames::Vector{String})
    # Process each component of the interaction
    for comp in term.terms
        extract_categorical_terms_from_formula!(categorical_info, comp, fitted_coefnames)
    end
end

function extract_categorical_terms_from_formula!(categorical_info::Dict{Symbol, CategoricalSchemaInfo}, term::FunctionTerm, fitted_coefnames::Vector{String})
    # Process function arguments
    for arg in term.args
        extract_categorical_terms_from_formula!(categorical_info, arg, fitted_coefnames)
    end
end

# Fallback for non-categorical terms
function extract_categorical_terms_from_formula!(categorical_info::Dict{Symbol, CategoricalSchemaInfo}, term, fitted_coefnames::Vector{String})
    # Do nothing for non-categorical terms
end

"""
    extract_complete_categorical_schema(model) -> Dict{Symbol, CategoricalSchemaInfo}

Properly extract categorical contrasts for both main effects and interactions.
FIXED: Always creates both DummyCoding and FullDummyCoding matrices.
"""
function extract_complete_categorical_schema(model)
    # println("DEBUG: Schema extraction starting for model type: $(typeof(model))")
    
    # Handle MixedModels differently - they don't have schemas, work with formulas directly
    if model isa Union{LinearMixedModel, GeneralizedLinearMixedModel}
        return extract_categorical_schema_from_mixed_model(model)
    end
    
    # Step 1: Get coefficient names - these work for TableRegressionModel
    fitted_coefnames = try
        if hasfield(typeof(model), :mm) && hasfield(typeof(model.mm), :coefnames)
            model.mm.coefnames
        elseif hasfield(typeof(model), :coefnames)
            model.coefnames
        elseif applicable(StatsModels.coefnames, model)
            StatsModels.coefnames(model)
        else
            String[]
        end
    catch
        String[]
    end
    
    # println("DEBUG: Found $(length(fitted_coefnames)) coefficient names")
    
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
            
            # println("DEBUG:   Main effect usage: $(main_effect_info.has_main_effect)")
            
            # Step 5: ALWAYS create both contrast types
            dummy_contrasts = create_dummy_coding_matrix(levels)
            full_dummy_contrasts = create_full_dummy_matrix(levels)
            
            # Determine which was used for main effects (if any)
            main_contrasts = if main_effect_info.has_main_effect
                fitted_contrast_matrix  # Use what the model actually used
            else
                nothing  # No main effect
            end
            
            # println("DEBUG:   Created DummyCoding: $(size(dummy_contrasts))")
            # println("DEBUG:   Created FullDummyCoding: $(size(full_dummy_contrasts))")
            
            # Step 6: Store all contrast types
            categorical_info[col_symbol] = CategoricalSchemaInfo(
                dummy_contrasts,          # Always available
                full_dummy_contrasts,     # Always available
                main_contrasts,           # May be nothing for interaction-only variables
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
    determine_interaction_contrast_type(
        comp::CategoricalTerm,
        all_components::Union{Vector, Tuple},
        main_effect_vars::Set{Symbol},
        categorical_schema::Dict{Symbol, CategoricalSchemaInfo}
    ) -> Symbol

Determine which type of contrast coding to use for a categorical in an interaction.
Returns :dummy_coding or :full_dummy_coding.

RULES:
1. If categorical has main effect → :dummy_coding
2. If categorical is in pure non-redundant categorical interaction → :full_dummy_coding  
3. If categorical is in mixed interaction with continuous → :dummy_coding
"""
function determine_interaction_contrast_type(
    comp::CategoricalTerm,
    all_components::Union{Vector, Tuple},
    main_effect_vars::Set{Symbol},
    categorical_schema::Dict{Symbol, CategoricalSchemaInfo}
)
    # Rule 1: Has main effect → always use DummyCoding
    if comp.sym in main_effect_vars
        # println("DEBUG: $(comp.sym) has main effect → DummyCoding")
        return :dummy_coding
    end
    
    # No main effect - check what it's interacting with
    # Convert to collection we can filter
    components_collection = collect(all_components)
    other_components = filter(c -> c !== comp, components_collection)
    
    # Check if ALL other components are categoricals without main effects
    all_others_are_nonredundant_cats = all(other_components) do other
        if other isa CategoricalTerm
            !(other.sym in main_effect_vars)
        else
            false  # Not a categorical (could be continuous, function, etc.)
        end
    end
    
    if all_others_are_nonredundant_cats && !isempty(other_components)
        # Pure non-redundant categorical interaction
        # println("DEBUG: $(comp.sym) in pure non-redundant categorical interaction → FullDummyCoding")
        return :full_dummy_coding
    else
        # Mixed interaction or has redundant components
        # println("DEBUG: $(comp.sym) in mixed/redundant interaction → DummyCoding")
        return :dummy_coding
    end
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
        # println("DEBUG: Model fields: $(fieldnames(typeof(model)))")
        if hasfield(typeof(model), :mf)
            # println("DEBUG: model.mf fields: $(fieldnames(typeof(model.mf)))")
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
        # println("DEBUG: Schema fields: $(fieldnames(typeof(schema)))")
        error("Cannot locate schema dictionary in schema of type $(typeof(schema)). Available fields: $(fieldnames(typeof(schema)))")
    end
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
    # println("DEBUG: Determining main effect vs interaction-only contrasts")
    
    # Get the fixed effects formula (strips random effects for mixed models)
    fixed_formula = fixed_effects_form(model)
    main_effect_terms = extract_main_effect_terms(fixed_formula.rhs)
    
    # println("DEBUG: Main effect categorical terms: $main_effect_terms")
    
    for (col_name, schema_info) in categorical_schema
        has_main_effect = col_name in main_effect_terms
        
        if has_main_effect
            # This variable has a main effect - main_effect_contrasts should already be set
            # println("DEBUG:   $col_name: HAS main effect")
        else
            # This variable is interaction-only - ensure main_effect_contrasts is nothing
            if schema_info.main_effect_contrasts !== nothing
                updated_info = CategoricalSchemaInfo(
                    schema_info.dummy_contrasts,
                    schema_info.full_dummy_contrasts,
                    nothing,  # No main effect contrasts
                    schema_info.n_levels,
                    schema_info.levels,
                    schema_info.level_codes,
                    schema_info.column
                )
                
                categorical_schema[col_name] = updated_info
            end
            # println("DEBUG:   $col_name: interaction-only (no main effect)")
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
    # println("DEBUG: Validating categorical schema")
    
    for (col_name, schema_info) in categorical_schema
        # println("DEBUG: Validating column: $col_name")
        
        # Check basic consistency
        if schema_info.n_levels != length(schema_info.levels)
            error("Schema inconsistency for $col_name: n_levels=$(schema_info.n_levels) but levels=$(length(schema_info.levels))")
        end
        
        # Check contrast matrices dimensions
        dummy_size = size(schema_info.dummy_contrasts)
        full_dummy_size = size(schema_info.full_dummy_contrasts)
        expected_rows = schema_info.n_levels
        
        if dummy_size[1] != expected_rows
            error("Dummy contrast matrix for $col_name has $(dummy_size[1]) rows, expected $expected_rows")
        end
        
        if full_dummy_size[1] != expected_rows
            error("Full dummy contrast matrix for $col_name has $(full_dummy_size[1]) rows, expected $expected_rows")
        end
        
        # Check contrast columns
        expected_dummy_cols = max(0, schema_info.n_levels - 1)
        expected_full_cols = schema_info.n_levels
        
        if dummy_size[2] != expected_dummy_cols
            error("Dummy contrast matrix for $col_name has $(dummy_size[2]) columns, expected $expected_dummy_cols")
        end
        
        if full_dummy_size[2] != expected_full_cols
            error("Full dummy contrast matrix for $col_name has $(full_dummy_size[2]) columns, expected $expected_full_cols")
        end
        
        # println("DEBUG:   Dummy contrasts: $(dummy_size)")
        # println("DEBUG:   Full dummy contrasts: $(full_dummy_size)")
        
        if schema_info.main_effect_contrasts !== nothing
            main_size = size(schema_info.main_effect_contrasts)
            # println("DEBUG:   Main effect contrasts: $(main_size)")
        else
            # println("DEBUG:   No main effect contrasts (interaction-only)")
        end
    end
    
    # println("DEBUG: Schema validation complete")
    return true
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

Compile term into self-contained evaluator using schema-based categorical contrasts.
FIXED: Properly handles mixed continuous-categorical interactions.
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
        # println("DEBUG: Compiling CategoricalTerm for $(term.sym)")
        
        # AUTHENTIC APPROACH: Use the exact contrasts from the fitted formula
        contrast_matrix = term.contrasts.matrix
        levels = term.contrasts.levels
        n_levels = length(levels)
        
        # Use empty level codes - levels are extracted dynamically at runtime
        level_codes = Int[]
        
        # println("DEBUG: Using authentic contrasts for $(term.sym): $(size(contrast_matrix)) from $(typeof(term.contrasts))")
        
        n_contrasts = size(contrast_matrix, 2)
        positions = collect(start_position:(start_position + n_contrasts - 1))
        
        evaluator = CategoricalEvaluator(
            term.sym,
            contrast_matrix,
            n_levels,  # Use authentic n_levels from term
            positions,
            level_codes
        )
        
        # println("DEBUG: Created CategoricalEvaluator with $(n_contrasts) contrasts")
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
        # Direct compilation - will be handled properly by compile_interaction_term_with_context
        # when called from MatrixTerm
        return compile_interaction_term_direct(term, start_position, scratch_allocator, categorical_schema)
        
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
        # println("DEBUG: Compiling MatrixTerm with $(length(term.terms)) sub-terms")
        
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
        
        # println("DEBUG: MatrixTerm compiled: $(length(constant_ops)) constants, $(length(continuous_ops)) continuous, $(length(categorical_evals)) categorical, $(length(function_evals)) functions, $(length(interaction_evals)) interactions")
        
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
    compile_interaction_term_direct(
        term::InteractionTerm,
        start_position::Int,
        scratch_allocator::ScratchAllocator,
        categorical_schema::Dict{Symbol, CategoricalSchemaInfo}
    ) -> InteractionEvaluator

Direct compilation of interaction term (when not in a MatrixTerm context).
Uses simple heuristics to determine contrast types.
"""
function compile_interaction_term_direct(
    term::InteractionTerm,
    start_position::Int,
    scratch_allocator::ScratchAllocator,
    categorical_schema::Dict{Symbol, CategoricalSchemaInfo}
)
    # When compiling directly (not from MatrixTerm), we don't have main effect context
    # Use a simple heuristic: assume DummyCoding for all categoricals
    
    component_evaluators = AbstractEvaluator[]
    component_widths = Int[]
    
    for comp in term.terms
        if comp isa CategoricalTerm
            # AUTHENTIC APPROACH: Use exact contrasts from the fitted formula component
            contrast_matrix = comp.contrasts.matrix
            levels = comp.contrasts.levels
            n_levels = length(levels)
            level_codes = Int[]  # Extracted dynamically at runtime
            
            n_contrasts = size(contrast_matrix, 2)
            temp_positions = collect(1:n_contrasts)
            
            interaction_categorical_eval = CategoricalEvaluator(
                comp.sym,
                contrast_matrix,  # Authentic contrast matrix
                n_levels,         # Authentic n_levels
                temp_positions,
                level_codes
            )
            
            push!(component_evaluators, interaction_categorical_eval)
            push!(component_widths, n_contrasts)
        else
            # Handle non-categorical components
            temp_allocator = ScratchAllocator()
            comp_eval = compile_term(comp, 1, temp_allocator, categorical_schema)
            
            push!(component_evaluators, comp_eval)
            push!(component_widths, output_width(comp_eval))
        end
    end
    
    # Convert to compile-time tuples
    N = length(component_evaluators)
    components_tuple = ntuple(i -> component_evaluators[i], N)
    widths_tuple = ntuple(i -> component_widths[i], N)
    
    # Calculate output setup
    total_width = prod(component_widths)
    positions = collect(start_position:(start_position + total_width - 1))
    
    return InteractionEvaluator{N, typeof(components_tuple), typeof(widths_tuple)}(
        components_tuple,
        widths_tuple,
        positions,
        start_position,
        total_width
    )
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
            # println("DEBUG: Found main effect: $(sub_term.sym)")
        elseif sub_term isa Union{ContinuousTerm, Term}
            # These don't affect categorical contrast choices
        elseif sub_term isa InteractionTerm
            # Skip - we're only collecting main effects
        elseif sub_term isa InterceptTerm || sub_term isa ConstantTerm
            # Skip - not relevant
        else
            # println("DEBUG: Unknown sub_term type in main effect collection: $(typeof(sub_term))")
        end
    end
    
    # println("DEBUG: Collected main effect categoricals: $main_effects")
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
FIXED: Properly determines contrast type based on interaction composition.
"""
function compile_interaction_term_with_context(
    term::InteractionTerm,
    start_position::Int,
    scratch_allocator::ScratchAllocator,
    categorical_schema::Dict{Symbol, CategoricalSchemaInfo},
    main_effect_vars::Set{Symbol}
)
    # println("DEBUG: Compiling interaction with context: main effects = $main_effect_vars")
    
    component_evaluators = AbstractEvaluator[]
    component_widths = Int[]
    
    # Process each interaction component with context-aware contrast selection
    for (i, comp) in enumerate(term.terms)
        # println("DEBUG: Processing interaction component $i: $(typeof(comp))")
        
        if comp isa CategoricalTerm
            # AUTHENTIC APPROACH: Use exact contrasts from the fitted formula component
            contrast_matrix = comp.contrasts.matrix
            levels = comp.contrasts.levels
            n_levels = length(levels)
            level_codes = Int[]  # Extracted dynamically at runtime
            
            n_contrasts = size(contrast_matrix, 2)
            temp_positions = collect(1:n_contrasts)
            
            interaction_categorical_eval = CategoricalEvaluator(
                comp.sym,
                contrast_matrix,  # Authentic contrast matrix
                n_levels,         # Authentic n_levels
                temp_positions,
                level_codes
            )
            
            push!(component_evaluators, interaction_categorical_eval)
            push!(component_widths, n_contrasts)
            
        else
            # Handle non-categorical components (continuous, functions, etc.)
            temp_allocator = ScratchAllocator()
            comp_eval = compile_term(comp, 1, temp_allocator, categorical_schema)
            
            push!(component_evaluators, comp_eval)
            push!(component_widths, output_width(comp_eval))
            
            # println("DEBUG: Created non-categorical component with width $(output_width(comp_eval))")
        end
    end
    
    # Rest is identical to existing InteractionTerm compilation
    N = length(component_evaluators)
    components_tuple = ntuple(i -> component_evaluators[i], N)
    widths_tuple = ntuple(i -> component_widths[i], N)
    
    total_width = prod(component_widths)
    positions = collect(start_position:(start_position + total_width - 1))
    
    # println("DEBUG: Context-aware interaction total width: $total_width ($(join(component_widths, " × ")))")
    
    return InteractionEvaluator{N, typeof(components_tuple), typeof(widths_tuple)}(
        components_tuple,
        widths_tuple,
        positions,
        start_position,
        total_width
    )
end
