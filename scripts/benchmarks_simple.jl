# Minimal configurable runner for FormulaCompiler benchmarks
# Edit the options below and run:
#   julia --project=. scripts/benchmarks_simple.jl

using Dates

# -----------------
# Options (edit me)
# -----------------
selected = [:core, :alloc, :scenario, :deriv, :margins, :se, :scale_n] # choose from: :core :alloc :scenario :deriv :margins :se :mixed :scaling :mixtures
fast = false # true for smaller n (quick iteration)
out = :md # :md, :csv, or nothing (e.g., out = nothing)
file = "" # output path; leave empty for auto under results/
tag = "" # optional label appended to filename and CSV column

# -----------------
# Runner (do not edit below unless needed)
# -----------------
include(joinpath(@__DIR__, "benchmarks.jl"))

results = run_selected(selected; fast=fast)

if out isa Symbol
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    if isempty(file)
        mkpath("results")
        tagpart = isempty(tag) ? "" : "_" * tag
        ext = out === :md ? ".md" : ".csv"
        file = joinpath("results", "benchmarks_" * ts * tagpart * ext)
    end
    if out === :md
        emit_markdown(file, results; tag=tag)
        println("\nWrote Markdown results to ", file)
    elseif out === :csv
        emit_csv(file, results; tag=tag)
        println("\nWrote CSV results to ", file)
    else
        @warn "Unknown out format" out
    end
end

