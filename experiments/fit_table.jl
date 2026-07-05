#=
fit_table.jl

Companion to §6 of the paper. Regenerates all three benchmark tables
from the bundled example panel of TruncatedDistributions.jl:

  Table A (§6.2): gradient-source comparison. Joint L-BFGS driven by
      (i)   finite-difference gradients on a generic cubature loss,
      (ii)  forward-mode AD (ForwardDiff.jl) on the same cubature loss,
      (iii) the closed-form Theorem-1 gradient via Kan–Robotti with the
            Genz–Bretz QMC base case,
      plus PRIMA NEWUOA (derivative-free) on the same loss as (iii).

  Table B (§6.3): warm-start ablation. LBFGS-explicit started from
      (μ̂, Σ̂) versus from the coordinate warm start; the warm-start
      column's wall-clock includes the warm start itself.

  Table C (§6.4): hybrid BCD versus warm-started joint L-BFGS across
      the full panel n = 2, …, 10. BCD settings follow the paper:
      greedy selection + full-loss acceptance at n ≤ 5; softmax
      selection + marginal-loss acceptance at n ≥ 6; joint L-BFGS
      polish at n = 6; marginal-loss endpoint (no polish) at n ≥ 7.

Protocol (§6.1): targets are the example's truncated moments rounded
to 1 decimal — computed by the Kan–Robotti recursion at n ≤ 6 and by
rejection sampling (2·10^5 accepted draws, seeded) at n ≥ 7. Caps:
50 outer iterations, 60 s wall-clock, loss target L < 10^-3. All runs
are single-threaded.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/fit_table.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using Distributions
using ForwardDiff
using ADTypes: AutoForwardDiff
using Optim
using Optim.NLSolversBase: only_fg!   # Optim 2 no longer re-exports it
using PRIMA
using LinearAlgebra
using MvNormalCDF
using Printf
using Random
using Statistics

BLAS.set_num_threads(1)

const FTARGET   = 1e-3
const MAX_ITERS = 50
const TIME_CAP  = 60.0
const COMMON_OPTS = Optim.Options(show_trace = false,
                                  iterations = MAX_ITERS,
                                  time_limit = TIME_CAP,
                                  callback   = s -> s.f_x < FTARGET)

# ----------------------------------------------------------------------
# Panel definition: (row label, n, bundled index, short description)
# ----------------------------------------------------------------------

struct PanelRow
    label::String
    n::Int
    index::Int
    descr::String
end

const PANEL = [
    PanelRow("2.1", 2, 1, "box"),
    PanelRow("2.2", 2, 2, "box"),
    PanelRow("2.3", 2, 3, "M.W. semi-inf."),
    PanelRow("2.4", 2, 4, "box"),
    PanelRow("3.1", 3, 1, "cube"),
    PanelRow("3.2", 3, 2, "semi-inf."),
    PanelRow("4.1", 4, 1, "box"),
    PanelRow("5.1", 5, 1, "[-2,2]^5"),
    PanelRow("5.2", 5, 2, "[-1.5,1.5]^5"),
    PanelRow("6.1", 6, 1, "[-2,2]^6"),
    PanelRow("6.2", 6, 2, "[-1.5,1.5]^6"),
    PanelRow("7.1", 7, 1, "[-2,2]^7"),
    PanelRow("7.2", 7, 2, "[-1.5,1.5]^7"),
    PanelRow("8.1", 8, 1, "[-2,2]^8"),
    PanelRow("8.2", 8, 2, "[-1.5,1.5]^8"),
    PanelRow("9.1", 9, 1, "[-2,2]^9"),
    PanelRow("9.2", 9, 2, "[-1.5,1.5]^9"),
    PanelRow("10.1", 10, 1, "[-2,2]^10"),
    PanelRow("10.2", 10, 2, "[-1.5,1.5]^10"),
]

# ----------------------------------------------------------------------
# Targets. KR moments at n ≤ 6; rejection sampling at n ≥ 7. Rounded to
# 1 decimal in both cases (MC standard error ≈ 2·10^-3 with 2·10^5
# accepted draws — far below the rounding granularity).
# ----------------------------------------------------------------------

const REJECTION_N   = 200_000
const REJECTION_MAX = 5_000_000

function rejection_moments(ex; rng = Random.Xoshiro(0))
    base = MvNormal(collect(Float64, ex.μ), Symmetric(Matrix{Float64}(ex.Σ)))
    a = collect(Float64, ex.a); b = collect(Float64, ex.b)
    n = length(a)
    kept = Matrix{Float64}(undef, n, REJECTION_N)
    got, tried = 0, 0
    while got < REJECTION_N
        tried >= REJECTION_MAX && error("rejection sampling budget exceeded")
        x = rand(rng, base)
        tried += 1
        if all(a .<= x .<= b)
            got += 1
            kept[:, got] = x
        end
    end
    return vec(mean(kept; dims = 2)), cov(kept; dims = 2)
end

function target_moments(row::PanelRow; digits::Int = 1)
    ex = get_example(n = row.n, index = row.index)
    if row.n <= 6
        set_kr_base_backend!(:mvnormalcdf)
        d = dist_from_example(ex)
        μ̂ = round.(collect(mean(d)); digits = digits)
        Σ̂ = round.(Matrix(cov(d));   digits = digits)
        p_mass = tp(d)
    else
        μ_raw, Σ_raw = rejection_moments(ex)
        μ̂ = round.(μ_raw; digits = digits)
        Σ̂ = round.(Σ_raw; digits = digits)
        p_mass = NaN     # reported from a dedicated QMC call instead
    end
    a = collect(Float64, ex.a); b = collect(Float64, ex.b)
    return μ̂, Σ̂, a, b, p_mass
end

# m^(0) via Genz–Bretz QMC directly (cheap at any n here).
function box_mass(row::PanelRow)
    ex = get_example(n = row.n, index = row.index)
    d = MvNormal(collect(Float64, ex.μ), Symmetric(Matrix{Float64}(ex.Σ)))
    return MvNormalCDF.mvnormcdf(d, collect(Float64, ex.a),
                                 collect(Float64, ex.b); m = 20_000)[1]
end

# ----------------------------------------------------------------------
# Generic cubature loss (type-generic; used by the FD and AD variants).
# Ported from the legacy benchmark: hcubature_inf handles ±Inf faces by
# tanh-type substitution; rtol = 1e-5.
# ----------------------------------------------------------------------

function unpack_generic(p::AbstractVector{T}, n::Int) where {T}
    μ = p[1:n]
    U = zeros(T, n, n)
    k = n + 1
    for i in 1:n, j in i:n
        U[i, j] = p[k]
        k += 1
    end
    return μ, U
end

function vector_moment_loss_hcub(p::AbstractVector{T}, a, b,
                                 μ̂, Σ̂; rtol = 1e-5,
                                 maxevals = 10_000_000) where {T}
    n = length(μ̂)
    μ, U = unpack_generic(p, n)
    Σinv = U' * U
    logdetΣ = -2 * sum(log, abs.(diag(U)))
    dens(x) = exp(-T(0.5) * (n * log(2π) + logdetΣ +
                             dot(x .- μ, Σinv * (x .- μ))))
    m0 = hcubature_inf(dens, a, b; rtol = rtol, maxevals = maxevals)[1]
    m1 = [hcubature_inf(x -> x[i] * dens(x), a, b;
                        rtol = rtol, maxevals = maxevals)[1] for i in 1:n]
    m2 = [hcubature_inf(x -> x[i] * x[j] * dens(x), a, b;
                        rtol = rtol, maxevals = maxevals)[1]
          for i in 1:n, j in 1:n]
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'
    return T(0.5) * (sum(abs2, μA .- μ̂) + sum(abs2, ΣA .- Σ̂))
end

# ----------------------------------------------------------------------
# The five fitting variants. Each takes a packed p0 and returns
# (final loss, wall seconds, converged-to-target flag).
# ----------------------------------------------------------------------

function fit_fd(p0, a, b, μ̂, Σ̂)
    f(p) = vector_moment_loss_hcub(p, a, b, μ̂, Σ̂)
    t = @elapsed res = optimize(f, p0, LBFGS(), COMMON_OPTS)
    return res.minimum, t
end

function fit_ad(p0, a, b, μ̂, Σ̂)
    f(p) = vector_moment_loss_hcub(p, a, b, μ̂, Σ̂)
    t = @elapsed res = optimize(f, p0, LBFGS(), COMMON_OPTS;
                                autodiff = AutoForwardDiff())
    return res.minimum, t
end

function fit_explicit(p0, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Matrix(Σ̂))
    t = @elapsed res = optimize(only_fg!(fg!), p0, LBFGS(), COMMON_OPTS)
    return res.minimum, t
end

function fit_newuoa(p0, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf)
    f(p) = vector_moment_loss(p, a, b, μ̂, Matrix(Σ̂))
    t = @elapsed begin
        p_final, _ = newuoa(f, p0; ftarget = FTARGET, maxfun = 5000)
        L_final = vector_moment_loss(p_final, a, b, μ̂, Matrix(Σ̂))
    end
    return L_final, t
end

# Warm-started packed initial point; returns (p0_ws, warm-start seconds).
function warmstart_p0(a, b, μ̂, Σ̂)
    t = @elapsed begin
        μ_ws, Σ_ws = warm_start_diagonal(μ̂, Σ̂, a, b)
        p0 = make_param_vec_from_μ_Σ(μ_ws, Σ_ws)
    end
    return p0, t
end

# Hybrid BCD through the public front door, settings per the paper.
function fit_bcd(row::PanelRow, a, b, μ̂, Σ̂)
    n = row.n
    selection = n <= 5 ? :greedy : :softmax
    accept_by = n <= 5 ? :full   : :marginal
    polish    = n == 6
    t = @elapsed begin
        _, _, info = fit_mvnormal(μ̂, Σ̂, a, b;
                                  method = :bcd,
                                  ftarget = FTARGET,
                                  bcd_selection = selection,
                                  bcd_accept_by = accept_by,
                                  bcd_polish = polish,
                                  verbose = false)
    end
    # info.loss: full loss with :full acceptance (n ≤ 5) and whenever the
    # polish branch ran (n = 6); the marginal-loss endpoint at n ≥ 7,
    # flagged with a dagger in the table.
    return info.loss, t, info
end

# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------

fmt_t(t) = isnan(t) ? "--" : @sprintf("%.2f", t)
fmt_L(L) = isnan(L) ? "--" : @sprintf("%.1e", L)
cell(L, t) = (isnan(t) ? "--" : @sprintf("%s / %s", fmt_t(t), fmt_L(L)))

# ----------------------------------------------------------------------
# Warmup: run every variant once on the cheapest example so the first
# timed row is post-JIT.
# ----------------------------------------------------------------------

function warmup()
    row = PanelRow("wu", 2, 4, "warmup")
    μ̂, Σ̂, a, b, _ = target_moments(row)
    p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
    for fit in (fit_fd, fit_ad, fit_explicit, fit_newuoa)
        try; fit(p0, a, b, μ̂, Σ̂); catch e; @warn "warmup" fit e; end
    end
    try; warmstart_p0(a, b, μ̂, Σ̂); catch; end
    try; fit_bcd(row, a, b, μ̂, Σ̂); catch e; @warn "warmup bcd" e; end
    return
end

# ----------------------------------------------------------------------
# Table A: gradient sources + NEWUOA, rows n ≤ 5.
# FD and AD are attempted at n ≤ 4 only (a single cubature loss
# evaluation at n = 5 exceeds the budget); NEWUOA at n ≤ 5.
# ----------------------------------------------------------------------

function tableA()
    println("\n" * "="^78)
    println("Table A: gradient-source comparison (LBFGS-FD / LBFGS-AD / LBFGS-explicit / NEWUOA)")
    println("="^78)
    latex = String[]
    for row in filter(r -> r.n <= 5, PANEL)
        μ̂, Σ̂, a, b, _ = target_moments(row)
        pm = box_mass(row)
        p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
        L_fd = t_fd = L_ad = t_ad = NaN
        if row.n <= 4
            try; L_fd, t_fd = fit_fd(p0, a, b, μ̂, Σ̂); catch; end
            try; L_ad, t_ad = fit_ad(p0, a, b, μ̂, Σ̂); catch; end
        end
        L_ex, t_ex = fit_explicit(p0, a, b, μ̂, Σ̂)
        L_nw = t_nw = NaN
        try; L_nw, t_nw = fit_newuoa(p0, a, b, μ̂, Σ̂); catch; end
        @printf("%-5s m0=%.2f  FD %s  AD %s  EXP %s  NEWUOA %s\n",
                row.label, pm, cell(L_fd, t_fd), cell(L_ad, t_ad),
                cell(L_ex, t_ex), cell(L_nw, t_nw))
        push!(latex, @sprintf("%s & %s & %.2f & %s & %s & %s & %s \\\\",
                              row.label, row.descr, pm,
                              cell(L_fd, t_fd), cell(L_ad, t_ad),
                              cell(L_ex, t_ex), cell(L_nw, t_nw)))
        flush(stdout)
    end
    println("\nLaTeX rows (Table A):")
    foreach(println, latex)
end

# ----------------------------------------------------------------------
# Table B: warm-start ablation, rows n ≤ 5.
# ----------------------------------------------------------------------

function tableB()
    println("\n" * "="^78)
    println("Table B: coordinate warm-start ablation (LBFGS-explicit vs +WS)")
    println("="^78)
    latex = String[]
    for row in filter(r -> r.n <= 5, PANEL)
        μ̂, Σ̂, a, b, _ = target_moments(row)
        pm = box_mass(row)
        p0_plain = make_param_vec_from_μ_Σ(μ̂, Σ̂)
        L0, t0 = fit_explicit(p0_plain, a, b, μ̂, Σ̂)
        p0_ws, t_ws = warmstart_p0(a, b, μ̂, Σ̂)
        L1, t1 = fit_explicit(p0_ws, a, b, μ̂, Σ̂)
        t1_tot = t1 + t_ws
        @printf("%-5s m0=%.2f  plain %s  +WS %s (ws %.3fs)\n",
                row.label, pm, cell(L0, t0), cell(L1, t1_tot), t_ws)
        push!(latex, @sprintf("%s & %s & %.2f & %s & %s \\\\",
                              row.label, row.descr, pm,
                              cell(L0, t0), cell(L1, t1_tot)))
        flush(stdout)
    end
    println("\nLaTeX rows (Table B):")
    foreach(println, latex)
end

# ----------------------------------------------------------------------
# Table C: hybrid BCD vs warm-started joint LBFGS, full panel.
# The +WS baseline is attempted through n = 6 (measured, not assumed)
# and dashed at n ≥ 7 where a single joint gradient call approaches or
# exceeds the wall-clock cap.
# ----------------------------------------------------------------------

function tableC()
    println("\n" * "="^78)
    println("Table C: hybrid BCD vs LBFGS-explicit+WS, full panel")
    println("="^78)
    latex = String[]
    for row in PANEL
        μ̂, Σ̂, a, b, _ = target_moments(row)
        pm = box_mass(row)
        L_ws = t_ws_tot = NaN
        if row.n <= 6
            p0_ws, t_ws = warmstart_p0(a, b, μ̂, Σ̂)
            L_ws, t_lb = fit_explicit(p0_ws, a, b, μ̂, Σ̂)
            t_ws_tot = t_lb + t_ws
        end
        L_bcd, t_bcd, info = fit_bcd(row, a, b, μ̂, Σ̂)
        marg = row.n >= 7 ? "*" : ""
        @printf("%-5s m0=%.2f  +WS %s  BCD %s%s  (iters=%d%s)\n",
                row.label, pm, cell(L_ws, t_ws_tot), cell(L_bcd, t_bcd),
                marg, info.iterations,
                get(info, :polished, false) ? ", polished" : "")
        push!(latex, @sprintf("%s & %s & %.2f & %s & %s%s \\\\",
                              row.label, row.descr, pm,
                              cell(L_ws, t_ws_tot), cell(L_bcd, t_bcd),
                              marg == "*" ? "\\,\$^{\\ddagger}\$" : ""))
        flush(stdout)
    end
    println("\nLaTeX rows (Table C):")
    foreach(println, latex)
end

# ----------------------------------------------------------------------

function main()
    which = isempty(ARGS) ? ["A", "B", "C"] : ARGS
    println("Panel targets: KR moments (n ≤ 6) / rejection sampling (n ≥ 7), rounded to 1 decimal.")
    println("Caps: $(MAX_ITERS) iterations, $(TIME_CAP) s, target L < $(FTARGET).")
    @info "warmup (JIT)..."
    warmup()
    "A" in which && tableA()
    "B" in which && tableB()
    "C" in which && tableC()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
