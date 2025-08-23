Validate locally

- Quick smoke: julia --project test/quick_test.jl
- Survey:
    - julia --project="." test/allocation_survey.jl
    - open test/allocation_results.csv.