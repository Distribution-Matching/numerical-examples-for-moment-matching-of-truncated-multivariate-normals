#=
t_fit_table.jl

Companion to the paper's Student-t benchmark. Moment matching for the
box-truncated multivariate t_ν with fixed ν on the panel rows 2.1–4.1 of
the Gaussian tables, at ν ∈ {4, 10}. Targets are the truncated mean and
covariance of t_ν(μ, Σ) for the example's (μ, Σ, A), computed by the
truncated-t recursion on an adaptive-cubature base case and rounded to one
decimal, as in the Gaussian tables. Every method optimises the same loss
(γ = 1), with the same caps: 50 outer iterations, 60 s, target L < 1e-3.

  LBFGS-FD       L-BFGS with finite-difference gradients on the loss
                 evaluated by the truncated-t recursion with an adaptive-
                 cubature base case (smooth, deterministic).
  NEWUOA         derivative-free (PRIMA) on the loss evaluated by the
                 recursion with the Genz–Bretz QMC base case.
  LBFGS-explicit L-BFGS with the closed-form gradient (t_ν moments of
                 order ≤ 2 and t_{ν+2} moments of order ≤ 4), QMC base case.
  ... +WS        the same, from the coordinate warm start (time included).

The naive start is (μ̂, (ν−2)/ν · Σ̂), which maps the target covariance to
a scale matrix.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_fit_table.jl
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
const MAX_ITERS = 50
const TIME_CAP = 60.0
const SEED = 20261015
const NUS = (4.0, 10.0)
const PANEL = [("2.1", 2, 1, "box"), ("2.2", 2, 2, "box"),
               ("2.3", 2, 3, "M.W. semi-inf."), ("2.4", 2, 4, "box"),
               ("3.1", 3, 1, "cube"), ("3.2", 3, 2, "semi-inf."),
               ("4.1", 4, 1, "box")]
const OPTS = Optim.Options(show_trace = false, iterations = MAX_ITERS,
                           time_limit = TIME_CAP, callback = s -> s.f_x < FTARGET)

reseed!() = set_kr_base_rng!(MersenneTwister(SEED))

function targets(ν, n, idx)
    ex = get_example(n = n, index = idx)
    a = collect(Float64, ex.a); b = collect(Float64, ex.b)
    set_kr_base_backend!(:hcubature)
    d = TruncatedMvT(ν, collect(Float64, ex.μ), Matrix{Float64}(ex.Σ), a, b;
                     max_moment_levels = 2)
    μ̂ = round.(mean(d); digits = 1)
    Σ̂ = round.(Matrix(cov(d)); digits = 1)
    return μ̂, Σ̂, a, b, tp(d)
end

naive_p0(ν, μ̂, Σ̂) = make_param_vec_from_μ_Σ(μ̂, ((ν - 2) / ν) .* Σ̂)

function fit_fd(ν, p0, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:hcubature)
    f(p) = vector_moment_loss_t(p, ν, a, b, μ̂, Σ̂)
    t = @elapsed res = optimize(f, p0, LBFGS(), OPTS)
    return res.minimum, t
end

struct TimeUp <: Exception end

# NEWUOA has no wall-clock cap of its own; enforce the common 60 s cap by
# aborting from inside the objective and reporting the best loss seen.
function fit_newuoa(ν, p0, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf); reseed!()
    best = Ref(Inf)
    t0 = time()
    function f(p)
        time() - t0 > TIME_CAP && throw(TimeUp())
        L = vector_moment_loss_t(p, ν, a, b, μ̂, Σ̂)
        best[] = min(best[], L)
        return L
    end
    t = @elapsed try
        newuoa(f, p0; ftarget = FTARGET, maxfun = 5000)
    catch e
        e isa TimeUp || rethrow()
    end
    return best[], t
end

function fit_explicit(ν, p0, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf); reseed!()
    fg!(F, G, p) = vector_fg_true_loss_t(F, G, p, ν, a, b, μ̂, Σ̂)
    t = @elapsed res = optimize(only_fg!(fg!), p0, LBFGS(), OPTS)
    return res.minimum, t
end

function fit_explicit_ws(ν, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf); reseed!()
    t = @elapsed begin
        μ0, Σ0 = warm_start_diagonal_t(ν, μ̂, Σ̂, a, b)
        L, _ = fit_explicit(ν, make_param_vec_from_μ_Σ(μ0, Σ0), a, b, μ̂, Σ̂)
    end
    return L, t
end

cell(L, t) = isnan(t) ? "--" : @sprintf("%.2f / %.1e", t, L)
texcell(L, t) = isnan(t) ? "--" :
    replace(@sprintf("%.2f / \$%.1e\$", t, L), r"e-0?(\d+)" => s"\\mathrm{e}\\text{-}\1")

function main()
    set_loss_gamma!(1.0)
    # JIT warm-up on the cheapest row.
    let (μ̂, Σ̂, a, b, _) = targets(10.0, 2, 4), p0 = naive_p0(10.0, μ̂, Σ̂)
        fit_fd(10.0, p0, a, b, μ̂, Σ̂); fit_newuoa(10.0, p0, a, b, μ̂, Σ̂)
        fit_explicit(10.0, p0, a, b, μ̂, Σ̂); fit_explicit_ws(10.0, a, b, μ̂, Σ̂)
    end
    latex = String[]
    println(@sprintf("%-5s %5s %6s  %-20s %-20s %-20s %-20s", "ex.", "ν", "m0",
                     "LBFGS-FD", "NEWUOA", "LBFGS-explicit", "LBFGS-explicit+WS"))
    for ν in NUS, (id, n, idx, descr) in PANEL
        μ̂, Σ̂, a, b, m0 = targets(ν, n, idx)
        p0 = naive_p0(ν, μ̂, Σ̂)
        r_fd = try fit_fd(ν, p0, a, b, μ̂, Σ̂) catch e; @warn "FD" id e; (NaN, NaN) end
        r_nw = try fit_newuoa(ν, p0, a, b, μ̂, Σ̂) catch e; @warn "NEWUOA" id e; (NaN, NaN) end
        r_ex = try fit_explicit(ν, p0, a, b, μ̂, Σ̂) catch e; @warn "EXP" id e; (NaN, NaN) end
        r_ws = try fit_explicit_ws(ν, a, b, μ̂, Σ̂) catch e; @warn "WS" id e; (NaN, NaN) end
        println(@sprintf("%-5s %5.1f %6.2f  %-20s %-20s %-20s %-20s", id, ν, m0,
                         cell(r_fd...), cell(r_nw...), cell(r_ex...), cell(r_ws...)))
        flush(stdout)
        push!(latex, @sprintf("%s & %s & %g & %.2f & %s & %s & %s & %s \\\\", id, descr,
                              ν, m0, texcell(r_fd...), texcell(r_nw...),
                              texcell(r_ex...), texcell(r_ws...)))
    end
    println("\nLaTeX rows:")
    foreach(println, latex)
end

main()
