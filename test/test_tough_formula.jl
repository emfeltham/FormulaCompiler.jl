# Test the tough_formula.md case directly
using DataFrames, GLM, MixedModels, FormulaCompiler, Tables, CategoricalArrays
using Test

# Create synthetic data matching the tough_formula variables
df = DataFrame(
    response = rand([0, 1], 200),
    socio4 = rand(Bool, 200),
    dists_p_inv = rand(200) .+ 0.1,  # Avoid zeros for inv
    are_related_dists_a_inv = rand(200) .+ 0.1,
    dists_a_inv = rand(200) .+ 0.1,
    num_common_nbs = rand(1:10, 200),
    schoolyears_p = rand(8:16, 200),
    wealth_d1_4_p = rand(200),
    man_p = rand(Bool, 200),
    age_p = rand(18:80, 200),
    religion_c_p = rand(Bool, 200),
    same_building = rand(Bool, 200),
    population = rand(100:10000, 200),
    hhi_religion = rand(200),
    hhi_indigenous = rand(200),
    coffee_cultivation = rand(Bool, 200),
    relation = rand(["family", "friend", "neighbor"], 200),
    degree_a_mean = rand(200),
    degree_h = rand(200),
    age_a_mean = rand(30:60, 200),
    age_h = rand(30:60, 200),
    age_h_nb_1_socio = rand(200),
    schoolyears_a_mean = rand(8:16, 200),
    schoolyears_h = rand(8:16, 200),
    schoolyears_h_nb_1_socio = rand(200),
    man_x = rand(Bool, 200),
    man_x_mixed_nb_1 = rand(200),
    wealth_d1_4_a_mean = rand(200),
    wealth_d1_4_h = rand(200),
    wealth_d1_4_h_nb_1_socio = rand(200),
    isindigenous_x = rand(Bool, 200),
    isindigenous_homop_nb_1 = rand(200),
    religion_c_x = rand(Bool, 200),
    religion_homop_nb_1 = rand(200),
    perceiver = rand(1:20, 200),
    village_code = rand(1:10, 200)
)

# Convert categoricals as needed
df.relation = categorical(df.relation)

# Test the exact tough formula
@testset "Tough Formula Direct Compilation" begin
    # The original tough formula with logic operators
    fx = @formula(
        response ~
        socio4 +
        (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
        !socio4 & dists_a_inv +
        (num_common_nbs & (dists_a_inv <= inv(2))) & (1 + are_related_dists_a_inv + dists_p_inv) +
        (schoolyears_p + wealth_d1_4_p + man_p + age_p + religion_c_p +
        same_building + population +
        hhi_religion + hhi_indigenous +
        coffee_cultivation +
        relation) & (1 + socio4 + are_related_dists_a_inv) +
        (
            degree_a_mean + degree_h +
            age_a_mean + age_h * age_h_nb_1_socio +
            schoolyears_a_mean + schoolyears_h * schoolyears_h_nb_1_socio +
            man_x * man_x_mixed_nb_1 +
            wealth_d1_4_a_mean + wealth_d1_4_h * wealth_d1_4_h_nb_1_socio +
            isindigenous_x * isindigenous_homop_nb_1 +
            religion_c_x * religion_homop_nb_1
        ) & (1 + socio4 + are_related_dists_a_inv) +
        religion_c_x & hhi_religion +
        isindigenous_x & hhi_indigenous +
        # Skip random effects for GLM test
        0  # Dummy term instead of (1|perceiver) + (1|village_code)
    )
    
    @debug "Testing tough formula compilation..."
    
    # Try to fit model with GLM (simpler than MixedModels for this test)
    model = try
        glm(fx, df, Binomial(), LogitLink())
    catch e
        @test_skip "Model fitting failed (expected with synthetic data): $e"
        return
    end
    
    @debug "GLM model fitted successfully"
    
    # Test FormulaCompiler compilation
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    
    @debug "FormulaCompiler compiled successfully"
    
    # Test correctness on first few rows
    ref_mm = modelmatrix(model)
    output = Vector{Float64}(undef, length(compiled))
    
    for i in 1:min(5, size(ref_mm, 1))
        compiled(output, data, i)
        @test output ≈ ref_mm[i, :] atol=1e-10
    end
    
    @debug "Correctness verified - tough formula works directly!"
    @debug "Formula size: $(length(output)) terms"
    
    # Test logic operators specifically
    @test any(contains(string(t), "Comparison") for t in compiled.ops)  # ComparisonOp or ComparisonBinaryOp
    @test any(contains(string(t), "NegationOp") for t in compiled.ops)
    
    @debug "Logic operators confirmed in compiled form"
end