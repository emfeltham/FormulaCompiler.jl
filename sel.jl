# select_model_vars
# main model css

cts_vars = [
    # distance variables are log
    # :dists_a,
    # :dists_p,
    # :are_related_dists_a,
    :age_a_mean,
    :wealth_d1_4_h,
    :wealth_d1_4_a_mean,
    :age_h,
    :schoolyears_p,
    :wealth_d1_4_p,
    :age_p,
    :population,
    :pct_catholic,
    :pct_protestant,
    :pct_indigenous,
    :religion_homop_nb_1,
    :isindigenous_homop_nb_1,
    :wealth_d1_4_h_nb_1_socio,
    :age_h_nb_1_socio,
    :degree_a_mean, :degree_h,
    :degree_p,
    :schoolyears_a_mean,
    :schoolyears_h,
    :schoolyears_h_nb_1_socio,
    :man_x_mixed_nb_1
];

othervars = [
    #:dists_a_notinf
    #:dists_p_notinf
    #:are_related_dists_a_notinf
    :response
    :perceiver
    :alter1
    :alter2
    :village_code
    :relation
    :kin431
    :same_building
    :socio4
    :dists_a
    :dists_p
    :are_related_dists_a
    :man_x
    :religion_c_x
    :isindigenous_x
    :isindigenous_p
    :man_p
    :religion_c_p
    :coffee_cultivation
    :market
    :maj_catholic
    :maj_indigenous
    :num_common_nbs
];


selvars = union(othervars, cts_vars);
unique!(selvars);

select!(df, selvars);
dropmissing!(df, selvars);
df.relation = categorical(df.relation);

