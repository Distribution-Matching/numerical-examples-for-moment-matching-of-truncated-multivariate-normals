#=
bcd_trace.jl

Profiling of a single hybrid-BCD run for the paper: the exact path the
solver takes through candidate sub-blocks, with the accept/reject
decision and per-target score at each step.

`block_coord_descent` (reached through `fit_mvnormal`) returns
`info.picks`, a vector of `(k, S, accepted, score)` tuples --- one per
outer iteration. This script runs one bundled example, prints the trace
and a summary (accept/reject counts, block-size histogram, which
coordinates each accepted block touched), and emits LaTeX rows for the
paper's trace table.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/bcd_trace.jl [n index]
Defaults to n = 7, index = 2 (the heavy [-1.5,1.5]^7 box).
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics

BLAS.set_num_threads(1)

const SEED = 20260706

# Rejection-sampling targets (matching fit_table.jl for n >= 7).
const REJECTION_N   = 200_000
const REJECTION_MAX = 5_000_000

function rejection_moments(ex; rng = Random.Xoshiro(0))
    base = MvNormal(collect(Float64, ex.μ), Symmetric(Matrix{Float64}(ex.Σ)))
    a = collect(Float64, ex.a); b = collect(Float64, ex.b)
    n = length(a)
    kept = Matrix{Float64}(undef, n, REJECTION_N)
    got, tried = 0, 0
    while got < REJECTION_N
        tried >= REJECTION_MAX && error("rejection budget exceeded")
        x = rand(rng, base); tried += 1
        if all(a .<= x .<= b)
            got += 1; kept[:, got] = x
        end
    end
    return vec(mean(kept; dims = 2)), cov(kept; dims = 2)
end

function targets(ex; digits = 1)
    n = length(ex.μ)
    if n <= 6
        set_kr_base_backend!(:mvnormalcdf)
        d = dist_from_example(ex)
        return round.(collect(mean(d)); digits), round.(Matrix(cov(d)); digits)
    else
        μ, Σ = rejection_moments(ex)
        return round.(μ; digits), round.(Σ; digits)
    end
end

Sstr(S) = "\\{" * join(S, ",") * "\\}"
Splain(S) = "{" * join(S, ",") * "}"

function main(n::Int, idx::Int)
    set_kr_base_backend!(:mvnormalcdf)
    ex = get_example(n = n, index = idx)
    a = collect(Float64, ex.a); b = collect(Float64, ex.b)
    μ̂, Σ̂ = targets(ex)

    accept_by = n <= 5 ? :full : :marginal
    polish    = n == 6

    # γ = 1 (balanced) to match the paper's benchmark tables; the
    # released package default is γ = 0.2.
    _, _, info = fit_mvnormal(μ̂, Σ̂, a, b;
                              method = :bcd, ftarget = 1e-3,
                              gamma = 1.0,
                              bcd_selection = :softmax,
                              bcd_accept_by = accept_by,
                              bcd_polish = polish,
                              seed = SEED,
                              verbose = false)

    picks = info.picks
    # info.hist = [L0, L_after_1, ...]; with :full acceptance it is the
    # full n-dim loss, so hist[t+1] is the total loss after picks[t].
    hist = info.hist
    n_acc = count(p -> p[3], picks)
    n_rej = length(picks) - n_acc
    ksizes = [p[1] for p in picks if p[3]]           # accepted block sizes
    blk_hist = Dict(k => count(==(k), ksizes) for k in 1:3)
    touched = zeros(Int, n)
    for p in picks
        p[3] || continue
        for i in p[2]; touched[i] += 1; end
    end

    println("="^66)
    @printf("BCD trace  ---  n=%d, example %d  (accept_by=%s%s)\n",
            n, idx, accept_by, polish ? ", polish" : "")
    println("="^66)
    hasL = length(hist) == length(picks) + 1
    @printf("L0 = %.3e\n", hist[1])
    @printf("%-5s %-3s %-14s %-12s %-9s %s\n",
            "iter", "k", "block S", "score", "decision", "L (full)")
    println("-"^70)
    for (t, p) in enumerate(picks)
        (k, S, acc, sc) = p
        Lstr = hasL ? @sprintf("%.3e", hist[t+1]) : "--"
        @printf("%-5d %-3d %-14s %-12.3e %-9s %s\n",
                t, k, Splain(S), sc, acc ? "accept" : "reject", Lstr)
    end
    println("-"^70)
    @printf("iterations=%d  accepts=%d  rejects=%d\n", length(picks), n_acc, n_rej)
    @printf("accepted block sizes: k=1:%d  k=2:%d  k=3:%d\n", blk_hist[1], blk_hist[2], blk_hist[3])
    @printf("updates per coordinate: %s\n", string(touched))
    @printf("final loss (info.loss) = %.3e\n", info.loss)

    sci(x) = (e = floor(Int, log10(abs(x))); m = x / 10.0^e;
              @sprintf("%.1f\\mathrm{e}\\text{-}%d", m, -e))

    println("\nLaTeX rows (iter & S & k & score & decision & L):")
    for (t, p) in enumerate(picks)
        (k, S, acc, sc) = p
        dec = acc ? "\\checkmark" : "\$\\times\$"
        Lcell = hasL ? "\$" * sci(hist[t+1]) * "\$" : "--"
        @printf("%d & \$%s\$ & %d & \$%s\$ & %s & %s \\\\\n",
                t, Sstr(S), k, sci(sc), dec, Lcell)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    n   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 7
    idx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
    main(n, idx)
end
