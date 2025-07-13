# phase1_structure_analysis.jl
# Phase 1: Proper structure analysis for compositional compiler

using Tables
using CategoricalArrays: CategoricalValue, levelcode
using StatsModels
using StandardizedPredictors: ZScoredTerm

###############################################################################
# Term Analysis Types
###############################################################################

"""
Analysis result for a single term, containing everything needed for compilation.
"""
struct TermAnalysis
    term::AbstractTerm                    # Original StatsModels term
    start_position::Int                   # First column in row_vec
    width::Int                           # Number of columns this term produces
    columns_used::Vector{Symbol}         # Data columns this term reads
    term_type::Symbol                    # :constant, :continuous, :categorical, :function, :interaction, :zscore
    metadata::Dict{Symbol, Any}          # Type-specific information
end

"""
Complete analysis of a formula's structure.
"""
struct FormulaAnalysis
    terms::Vector{TermAnalysis}          # All terms in evaluation order
    total_width::Int                     # Total columns in model matrix
    all_columns::Vector{Symbol}          # All data columns used
    position_map::Dict{AbstractTerm, UnitRange{Int}}  # Term -> column range mapping
end

###############################################################################
# Phase 1: Structure Analysis
###############################################################################

"""
    analyze_formula_structure(model) -> FormulaAnalysis

Phase 1: Analyze the formula structure to understand term widths, positions,
and dependencies without generating any code.
"""
function analyze_formula_structure(model)
    rhs = fixed_effects_form(model).rhs
    
    println("=== Phase 1: Formula Structure Analysis ===")
    println("Formula RHS: $rhs")
    println("RHS type: $(typeof(rhs))")
    
    # Extract terms in StatsModels order
    terms = extract_terms_properly(rhs)
    println("Extracted $(length(terms)) terms:")
    for (i, term) in enumerate(terms)
        println("  $i: $(typeof(term)) - $term")
    end
    
    # Analyze each term
    term_analyses = TermAnalysis[]
    current_position = 1
    all_columns = Symbol[]
    position_map = Dict{AbstractTerm, UnitRange{Int}}()
    
    for (i, term) in enumerate(terms)
        println("\nAnalyzing term $i: $term")
        
        analysis = analyze_single_term(term, current_position)
        push!(term_analyses, analysis)
        
        # Update tracking
        append!(all_columns, analysis.columns_used)
        position_map[term] = current_position:(current_position + analysis.width - 1)
        current_position += analysis.width
        
        println("  → Type: $(analysis.term_type)")
        println("  → Width: $(analysis.width)")
        println("  → Position: $(analysis.start_position):$(analysis.start_position + analysis.width - 1)")
        println("  → Columns: $(analysis.columns_used)")
        if !isempty(analysis.metadata)
            println("  → Metadata: $(analysis.metadata)")
        end
    end
    
    total_width = current_position - 1
    unique_columns = unique(all_columns)
    
    println("\n=== Analysis Summary ===")
    println("Total width: $total_width")
    println("Unique columns: $unique_columns")
    
    # Validate against model matrix if available
    try
        mm = modelmatrix(model)
        expected_width = size(mm, 2)
        if total_width == expected_width
            println("✅ Width matches model matrix: $total_width")
        else
            println("⚠️  Width mismatch! Expected: $expected_width, Got: $total_width")
        end
    catch e
        println("⚠️  Could not validate against model matrix: $e")
    end
    
    return FormulaAnalysis(term_analyses, total_width, unique_columns, position_map)
end

###############################################################################
# Term Extraction (Preserve StatsModels Structure)
###############################################################################

"""
Extract terms in the correct order, preserving StatsModels structure.
This is crucial - don't decompose complex terms here.
"""
function extract_terms_properly(rhs::AbstractTerm)
    # Handle different RHS structures
    if rhs isa MatrixTerm
        return collect(rhs.terms)
    elseif rhs isa Tuple
        return collect(rhs)
    elseif rhs isa Vector
        return rhs
    else
        # Single term
        return [rhs]
    end
end

###############################################################################
# Single Term Analysis
###############################################################################

"""
Analyze a single term to determine its characteristics.
This is where we respect StatsModels term structure.
"""
function analyze_single_term(term::AbstractTerm, start_position::Int)
    if term isa InterceptTerm
        return analyze_intercept_term(term, start_position)
    elseif term isa ConstantTerm
        return analyze_constant_term(term, start_position)
    elseif term isa Union{ContinuousTerm, Term}
        return analyze_continuous_term(term, start_position)
    elseif term isa CategoricalTerm
        return analyze_categorical_term(term, start_position)
    elseif term isa FunctionTerm
        return analyze_function_term(term, start_position)
    elseif term isa InteractionTerm
        return analyze_interaction_term(term, start_position)
    elseif term isa ZScoredTerm
        return analyze_zscore_term(term, start_position)
    else
        @warn "Unknown term type: $(typeof(term)), treating as constant"
        return TermAnalysis(term, start_position, 1, Symbol[], :unknown, 
                          Dict(:fallback => true))
    end
end

function analyze_intercept_term(term::InterceptTerm, start_position::Int)
    if hasintercept(term)
        return TermAnalysis(term, start_position, 1, Symbol[], :constant,
                          Dict(:value => 1.0))
    else
        return TermAnalysis(term, start_position, 0, Symbol[], :constant,
                          Dict(:value => 0.0, :omitted => true))
    end
end

function analyze_constant_term(term::ConstantTerm, start_position::Int)
    return TermAnalysis(term, start_position, 1, Symbol[], :constant,
                      Dict(:value => Float64(term.n)))
end

function analyze_continuous_term(term::Union{ContinuousTerm, Term}, start_position::Int)
    return TermAnalysis(term, start_position, 1, [term.sym], :continuous,
                      Dict(:column => term.sym))
end

function analyze_categorical_term(term::CategoricalTerm, start_position::Int)
    # This is critical: get the actual width from the contrast matrix
    contrast_matrix = term.contrasts.matrix
    actual_width = size(contrast_matrix, 2)
    
    metadata = Dict{Symbol, Any}(
        :column => term.sym,
        :contrast_matrix => Matrix{Float64}(contrast_matrix),
        :n_levels => size(contrast_matrix, 1),
        :n_contrasts => actual_width
    )
    
    return TermAnalysis(term, start_position, actual_width, [term.sym], :categorical, metadata)
end

function analyze_function_term(term::FunctionTerm, start_position::Int)
    func = term.f
    args = term.args
    
    # Extract columns used by this function
    columns_used = extract_function_columns(term)
    
    metadata = Dict{Symbol, Any}(
        :function => func,
        :n_args => length(args),
        :args => args
    )
    
    # Analyze specific function patterns
    if length(args) == 1
        metadata[:pattern] = :unary
        if args[1] isa Union{ContinuousTerm, Term}
            metadata[:simple_unary] = true
            metadata[:arg_column] = args[1].sym
        end
    elseif length(args) == 2 && func === (^) && args[2] isa ConstantTerm
        metadata[:pattern] = :power
        metadata[:base_column] = args[1].sym
        metadata[:exponent] = Float64(args[2].n)
    elseif length(args) == 2
        metadata[:pattern] = :binary
    else
        metadata[:pattern] = :complex
    end
    
    # Function terms always produce 1 column
    return TermAnalysis(term, start_position, 1, columns_used, :function, metadata)
end

function analyze_interaction_term(term::InteractionTerm, start_position::Int)
    components = term.terms
    
    # Analyze each component to understand the interaction structure
    component_info = []
    all_columns = Symbol[]
    
    for comp in components
        comp_width = width(comp)
        comp_columns = extract_term_columns(comp)
        append!(all_columns, comp_columns)
        
        push!(component_info, Dict(
            :term => comp,
            :width => comp_width,
            :columns => comp_columns,
            :type => typeof(comp)
        ))
    end
    
    # Calculate total width as product of component widths
    component_widths = [info[:width] for info in component_info]
    total_width = prod(component_widths)
    
    metadata = Dict{Symbol, Any}(
        :components => component_info,
        :component_widths => component_widths,
        :total_width => total_width,
        :n_components => length(components)
    )
    
    return TermAnalysis(term, start_position, total_width, unique(all_columns), :interaction, metadata)
end

function analyze_zscore_term(term::ZScoredTerm, start_position::Int)
    # Analyze the underlying term
    underlying = term.term
    underlying_analysis = analyze_single_term(underlying, start_position)
    
    # ZScore terms have same width as underlying term
    metadata = Dict{Symbol, Any}(
        :underlying_term => underlying,
        :underlying_type => underlying_analysis.term_type,
        :center => term.center isa Number ? Float64(term.center) : Float64(term.center[1]),
        :scale => term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
    )
    
    return TermAnalysis(term, start_position, underlying_analysis.width, 
                      underlying_analysis.columns_used, :zscore, metadata)
end

###############################################################################
# Column Extraction Utilities
###############################################################################

"""
Extract all data columns used by a term, recursively.
"""
function extract_term_columns(term::AbstractTerm)
    if term isa Union{ContinuousTerm, Term}
        return [term.sym]
    elseif term isa CategoricalTerm
        return [term.sym]
    elseif term isa FunctionTerm
        return extract_function_columns(term)
    elseif term isa InteractionTerm
        columns = Symbol[]
        for comp in term.terms
            append!(columns, extract_term_columns(comp))
        end
        return unique(columns)
    elseif term isa ZScoredTerm
        return extract_term_columns(term.term)
    else
        return Symbol[]
    end
end

function extract_function_columns(term::FunctionTerm)
    columns = Symbol[]
    for arg in term.args
        if arg isa Union{ContinuousTerm, Term}
            push!(columns, arg.sym)
        elseif arg isa FunctionTerm
            append!(columns, extract_function_columns(arg))
        elseif arg isa InteractionTerm
            append!(columns, extract_term_columns(arg))
        # ConstantTerm contributes no columns
        end
    end
    return unique(columns)
end

###############################################################################
# Analysis Validation and Utilities
###############################################################################

"""
Validate the analysis against the actual model matrix.
"""
function validate_analysis(analysis::FormulaAnalysis, model)
    try
        mm = modelmatrix(model)
        expected_width = size(mm, 2)
        
        if analysis.total_width != expected_width
            @error "Width mismatch!" expected=expected_width got=analysis.total_width
            return false
        end
        
        # Check that position ranges don't overlap and cover all columns
        used_positions = Set{Int}()
        for term_analysis in analysis.terms
            range = term_analysis.start_position:(term_analysis.start_position + term_analysis.width - 1)
            for pos in range
                if pos in used_positions
                    @error "Position overlap detected at column $pos"
                    return false
                end
                push!(used_positions, pos)
            end
        end
        
        if length(used_positions) != analysis.total_width
            @error "Position coverage mismatch!" covered=length(used_positions) total=analysis.total_width
            return false
        end
        
        println("✅ Analysis validation passed")
        return true
        
    catch e
        @warn "Could not validate analysis: $e"
        return false
    end
end

"""
Get the term analysis for a specific position in the model matrix.
"""
function get_term_at_position(analysis::FormulaAnalysis, position::Int)
    for term_analysis in analysis.terms
        if term_analysis.start_position <= position <= term_analysis.start_position + term_analysis.width - 1
            return term_analysis
        end
    end
    return nothing
end

"""
Print detailed analysis summary.
"""
function print_analysis_summary(analysis::FormulaAnalysis)
    println("\n=== Detailed Analysis Summary ===")
    println("Total terms: $(length(analysis.terms))")
    println("Total width: $(analysis.total_width)")
    println("Data columns: $(analysis.all_columns)")
    
    println("\nTerm breakdown:")
    for (i, term) in enumerate(analysis.terms)
        range_str = if term.width == 1
            "$(term.start_position)"
        else
            "$(term.start_position):$(term.start_position + term.width - 1)"
        end
        println("  $i. $(term.term_type) [$range_str] $(term.term)")
        if !isempty(term.columns_used)
            println("     Uses columns: $(term.columns_used)")
        end
        if !isempty(term.metadata)
            for (key, value) in term.metadata
                if key != :contrast_matrix  # Don't print large matrices
                    println("     $key: $value")
                end
            end
        end
    end
end

###############################################################################
# Testing and Examples
###############################################################################

"""
Test the structure analysis on various formula types.
"""
function test_structure_analysis()
    println("=== Testing Structure Analysis ===")
    
    # Create test data
    Random.seed!(123)
    n = 20
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n))
    )
    
    # Test formulas of increasing complexity
    test_cases = [
        (@formula(y ~ 1), "Intercept only"),
        (@formula(y ~ x), "Simple continuous"),
        (@formula(y ~ group), "Simple categorical"),
        (@formula(y ~ x + group), "Mixed terms"),
        (@formula(y ~ x^2), "Power function"),
        (@formula(y ~ log(z)), "Log function"),
        (@formula(y ~ x * group), "Simple interaction"),
        (@formula(y ~ x^2 * log(z)), "Complex interaction"),
        (@formula(y ~ x + x^2 + log(z) + group + x*group), "Kitchen sink"),
        (@formula(y ~ x + x^2 + log(z) + group + x*group + (x>0) + log(z)*x), "Bigger Kitchen sink"),
        (@formula(y ~ x*z*group*x^2), "Interactive")
    ]
    
    successful = 0
    
    for (i, (formula, description)) in enumerate(test_cases)
        println("\n--- Test $i: $description ---")
        println("Formula: $formula")
        
        try
            model = lm(formula, df)
            analysis = analyze_formula_structure(model)
            
            if validate_analysis(analysis, model)
                println("✅ Analysis successful and validated")
                successful += 1
            else
                println("❌ Analysis validation failed")
            end
            
        catch e
            println("❌ Test failed: $e")
            @error "Test $i failed" exception=(e, catch_backtrace())
        end
    end
    
    println("\n=== Test Summary ===")
    println("Successful: $successful / $(length(test_cases))")
    
    return successful == length(test_cases)
end

function test_structure_analysis_standard()
    println("=== Testing Structure Analysis ===")
    
    # Create test data
    Random.seed!(123)
    n = 20
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n))
    )
    
    # Test formulas of increasing complexity
    test_cases = [
        (@formula(y ~ 1), "Intercept only"),
        (@formula(y ~ x), "Simple continuous"),
        (@formula(y ~ group), "Simple categorical"),
        (@formula(y ~ x + group), "Mixed terms"),
        (@formula(y ~ x^2), "Power function"),
        (@formula(y ~ log(z)), "Log function"),
        (@formula(y ~ x * group), "Simple interaction"),
        (@formula(y ~ x^2 * log(z)), "Complex interaction"),
        (@formula(y ~ x + x^2 + log(z) + group + x*group), "Kitchen sink"),
        (@formula(y ~ x + x^2 + log(z) + group + x*group + (x>0) + log(z)*x), "Bigger Kitchen sink"),
        (@formula(y ~ x*z*group*x^2), "Interactive")
    ]
        
    contrasts = Dict(:x => ZScore());

    successful = 0
    
    for (i, (formula, description)) in enumerate(test_cases)
        println("\n--- Test $i: $description ---")
        println("Formula: $formula")
        
        try
            model = lm(formula, df; contrasts)
            analysis = analyze_formula_structure(model)
            
            if validate_analysis(analysis, model)
                println("✅ Analysis successful and validated")
                successful += 1
            else
                println("❌ Analysis validation failed")
            end
            
        catch e
            println("❌ Test failed: $e")
            @error "Test $i failed" exception=(e, catch_backtrace())
        end
    end
    
    println("\n=== Test Summary ===")
    println("Successful: $successful / $(length(test_cases))")
    
    return successful == length(test_cases)
end


# Export main functions
export analyze_formula_structure, FormulaAnalysis, TermAnalysis
export validate_analysis, print_analysis_summary, test_structure_analysis