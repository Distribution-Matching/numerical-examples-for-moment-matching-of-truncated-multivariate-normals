#=
fit_table_closed.jl

The closed-form and derivative-free columns of the paper's normal fit
table on the panel rows 2.1–8.2: L-BFGS on the closed-form gradient from
the naive start (explicit) and from the coordinate warm start (+WS), and
NEWUOA on the recursion loss. Targets are the truncated mean and
covariance of each example, by the Kan–Robotti recursion (now affordable
at every n of the panel), rounded to one decimal. Loss weight γ = 1;
caps: 50 iterations, 60 s (enforced inside the objective for NEWUOA),
target L < 1e-3. The FD and AD columns (n ≤ 4) do not depend on the
recursion and come from fit_table.jl.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/fit_table_closed.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using Optim
using Optim.NLSolversBase: only_fg!
using PRIMA
using LinearAlgebra
using Printf
using Random

BLAS.set_num_threads(1)

const FTARGET = 1e-3
const TIME_CAP = 60.0
const SEED = 20260706
const PANEL = [("2.1", 2, 1), ("2.2", 2, 2), ("2.3", 2, 3), ("2.4", 2, 4),
               ("3.1", 3, 1), ("3.2", 3, 2), ("4.1", 4, 1),
               ("5.1", 5, 1), ("5.2", 5, 2), ("6.1", 6, 1), ("6.2", 6, 2),
               ("7.1", 7, 1), ("7.2", 7, 2), ("8.1", 8, 1), ("8.2", 8, 2)]
const OPTS = Optim.Options(show_trace = false, iterations = 50, time_limit = TIME_CAP,
                           callback = s -> s.f_x < FTARGET)

reseed!() = set_kr_base_rng!(MersenneTwister(SEED))

function targets(n, idx)
    ex = get_example(n = n, index = idx)
    set_kr_base_backend!(:mvnormalcdf); reseed!()
    a = collect(Float64, ex.a); b = collect(Float64, ex.b)
    d = RecursiveMomentsBoxTruncatedMvNormal(collect(Float64, ex.μ),
            TruncatedDistributions.PDMat(Matrix{Float64}(ex.Σ)), a, b; max_moment_levels = 2)
    m0 = raw_moment_from_indices(d, Int[])
    m1 = [raw_moment_from_indices(d, [i]) for i in 1:n] ./ m0
    m2 = [raw_moment_from_indices(d, [i, j]) for i in 1:n, j in 1:n] ./ m0
    return round.(m1; digits = 1), round.(m2 .- m1 * m1'; digits = 1), a, b, m0
end

function fit_explicit(p0, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf); reseed!()
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Σ̂)
    t = @elapsed res = optimize(only_fg!(fg!), p0, LBFGS(), OPTS)
    return res.minimum, t
end

function fit_ws(a, b, μ̂, Σ̂)
    t = @elapsed begin
        μ0, Σ0 = warm_start_diagonal(μ̂, Σ̂, a, b)
        L, _ = fit_explicit(make_param_vec_from_μ_Σ(μ0, Σ0), a, b, μ̂, Σ̂)
    end
    return L, t
end

struct TimeUp <: Exception end
function fit_newuoa(p0, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf); reseed!()
    best = Ref(Inf); t0 = time()
    f(p) = (time() - t0 > TIME_CAP && throw(TimeUp());
            L = vector_moment_loss(p, a, b, μ̂, Σ̂); best[] = min(best[], L); L)
    t = @elapsed try
        newuoa(f, p0; ftarget = FTARGET, maxfun = 5000)
    catch e
        e isa TimeUp || rethrow()
    end
    return best[], t
end

cell(L, t) = @sprintf("%.2f / %.1e", t, L)

function main()
    set_loss_gamma!(1.0)
    let (μ̂, Σ̂, a, b, _) = targets(2, 4), p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
        fit_explicit(p0, a, b, μ̂, Σ̂); fit_ws(a, b, μ̂, Σ̂); fit_newuoa(p0, a, b, μ̂, Σ̂)
    end
    println(@sprintf("%-5s %6s  %-18s %-18s %-18s", "ex.", "m0", "NEWUOA", "explicit", "+WS"))
    for (id, n, idx) in PANEL
        μ̂, Σ̂, a, b, m0 = targets(n, idx)
        p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
        rn = fit_newuoa(p0, a, b, μ̂, Σ̂)
        re = fit_explicit(p0, a, b, μ̂, Σ̂)
        rw = fit_ws(a, b, μ̂, Σ̂)
        println(@sprintf("%-5s %6.2f  %-18s %-18s %-18s", id, m0, cell(rn...), cell(re...),
                         cell(rw...)))
        flush(stdout)
    end
end

main()
