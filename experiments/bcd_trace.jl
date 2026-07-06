#=
bcd_trace.jl — data for the paper's BCD loss-trace figure.

Runs one complete hybrid-BCD fit and records, per outer iteration, the
sub-block S the softmax rule visited, its size, the accept/reject
decision, and the full loss L afterwards. Full-loss acceptance is used so
the loss curve is exactly the full n-dimensional loss and is monotone
(it steps down only at accepts). The run starts from the naive point
(μ̂, Σ̂) rather than the warm start, so the descent spans a visible range.

Writes `results/bcd_trace_data.txt` (header `# ...`, then rows
`iter S k accept L`), which `bcd_trace_plot.py` turns into the figure.

Run:  OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/bcd_trace.jl [n index]
Defaults to n = 6, index = 1.
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using LinearAlgebra
using Printf
using Statistics

BLAS.set_num_threads(1)

const SEED = 20260706

function main(n::Int, idx::Int)
    set_kr_base_backend!(:mvnormalcdf)
    ex = get_example(n = n, index = idx)
    a = collect(Float64, ex.a); b = collect(Float64, ex.b)
    d = dist_from_example(ex)
    μ̂ = round.(collect(mean(d)); digits = 1)
    Σ̂ = round.(Matrix(cov(d));   digits = 1)

    # Full-loss acceptance, softmax selection, naive start; γ = 1 to match
    # the benchmark tables.
    _, _, info = fit_mvnormal(μ̂, Σ̂, a, b;
                              method = :bcd, gamma = 1.0,
                              bcd_selection = :softmax,
                              bcd_accept_by = :full,
                              seed = SEED, iterations = 30,
                              μ_init = μ̂, Σ_init = Σ̂)
    picks = info.picks
    hist  = info.hist                  # [L0, L_after_1, ...]; full loss
    n_acc = count(p -> p[3], picks)

    outdir = joinpath(@__DIR__, "results")
    datafile = joinpath(outdir, "bcd_trace_data.txt")
    open(datafile, "w") do io
        @printf(io, "# n=%d index=%d iters=%d accepts=%d L0=%.6e Lf=%.6e\n",
                n, idx, length(picks), n_acc, hist[1], hist[end])
        @printf(io, "0 - 0 1 %.6e\n", hist[1])         # naive start
        for (t, p) in enumerate(picks)
            @printf(io, "%d %s %d %d %.6e\n",
                    t, join(p[2], ","), p[1], p[3] ? 1 : 0, hist[t + 1])
        end
    end

    @printf("BCD trace (n=%d, ex %d): %d iterations, %d accepted; L %.2e -> %.2e\n",
            n, idx, length(picks), n_acc, hist[1], hist[end])
    @printf("%-4s %-12s %-3s %-8s %s\n", "iter", "S", "k", "decision", "L")
    for (t, p) in enumerate(picks)
        @printf("%-4d %-12s %-3d %-8s %.3e\n",
                t, "{" * join(p[2], ",") * "}", p[1],
                p[3] ? "accept" : "reject", hist[t + 1])
    end
    println("wrote $datafile")
end

if abspath(PROGRAM_FILE) == @__FILE__
    n   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
    idx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    main(n, idx)
end
