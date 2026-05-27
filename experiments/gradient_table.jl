#=
gradient_table.jl

Companion to §4 of the paper: wall-clock and accuracy of the
gradient of the moment-matching loss L = L_1 + L_2 (eq. (12) in
the paper) computed three ways:

  (a) closed-form via Theorem 1 (TruncatedDistributions.grad_true_loss)
  (b) forward-mode automatic differentiation via ForwardDiff.jl
  (c) finite differences (one-sided, h = 1e-5)

For each (μ, Σ, A) we evaluate the gradient at θ = (μ + δμ,
chol(Σ + δΣ)^{-T}) where (δμ, δΣ) is a small perturbation, and
targets (μ̂, Σ̂) = (μ_A, Σ_A) at the underlying (μ, Σ) so the
gradient at the true θ is essentially zero. The perturbation
makes the gradient non-trivial without leaving the local basin.

Run from the repository root:
    julia --project=experiments experiments/gradient_table.jl

Implementation notes
- The closed-form path uses set_kr_base_backend!(:mvnormalcdf)
  (Genz–Bretz QMC, Float64-only).
- The AD and FD paths share an inner `loss(θ)` that constructs
  TruncatedMvNormal with the :hcubature backend so the integrand is
  type-generic (ForwardDiff.Dual-compatible).
- The accuracy column compares each method's gradient against the
  closed-form gradient; the FD column reports relative norm error,
  the AD column reports max relative error on the components.
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using Distributions
using ForwardDiff
using HCubature
using BenchmarkTools
using LinearAlgebra
using Printf
using Random

# ----------------------------------------------------------------------
# pack / unpack between θ vector and (μ, U)
# ----------------------------------------------------------------------

function pack_params(μ::AbstractVector, U::AbstractMatrix)
    n = length(μ)
    T = promote_type(eltype(μ), eltype(U))
    θ = Vector{T}(undef, n + n*(n+1)÷2)
    @inbounds begin
        θ[1:n] .= μ
        k = n
        for i in 1:n, j in i:n
            k += 1
            θ[k] = U[i, j]
        end
    end
    return θ
end

function unpack_params(θ::AbstractVector, n::Int)
    T = eltype(θ)
    μ = θ[1:n]
    U = zeros(T, n, n)
    @inbounds begin
        k = n
        for i in 1:n, j in i:n
            k += 1
            U[i, j] = θ[k]
        end
    end
    return μ, U
end

# ----------------------------------------------------------------------
# Loss L(θ) for use by AD and FD. Constructs TruncatedMvNormal with
# the :hcubature backend so the integrand is type-generic.
# ----------------------------------------------------------------------

# Direct type-generic loss. TruncatedMvNormal's constructor is not
# Dual-compatible (Float64-only internally), so for AD/FD we
# implement μ_A, Σ_A via direct HCubature on the explicit Gaussian
# density. Finite boxes only — semi-infinite axes must be clipped
# by the caller.
function gaussian_pdf(x, μ, Σ_inv, log_det_Σ, n)
    z = x .- μ
    Q = dot(z, Σ_inv * z)
    return exp(-0.5 * (n * log(2π) + log_det_Σ + Q))
end

function loss(θ::AbstractVector, n::Int, a::Vector{Float64},
              b::Vector{Float64}, μ̂::Vector{Float64},
              Σ̂::Matrix{Float64}; rtol::Float64 = 1e-8)
    μ, U = unpack_params(θ, n)
    Σ_inv = U' * U
    log_det_Σ = -2 * log(prod(U[i, i] for i in 1:n))   # log|Σ| = -log|U^TU|
    pdf = x -> gaussian_pdf(x, μ, Σ_inv, log_det_Σ, n)

    m0, _ = hcubature(pdf, a, b; rtol = rtol)
    μ_A = [hcubature(x -> x[i] * pdf(x), a, b; rtol = rtol)[1] / m0 for i in 1:n]
    Σ_A = Matrix{eltype(θ)}(undef, n, n)
    for i in 1:n, j in i:n
        mij, _ = hcubature(x -> x[i] * x[j] * pdf(x), a, b; rtol = rtol)
        Σ_A[i, j] = mij / m0 - μ_A[i] * μ_A[j]
        Σ_A[j, i] = Σ_A[i, j]
    end
    L1 = 0.5 * sum(abs2, μ_A .- μ̂)
    L2 = 0.5 * sum(abs2, Σ_A .- Σ̂)
    return L1 + L2
end

# ----------------------------------------------------------------------
# Three gradient methods. Each returns a Vector{Float64} packed in
# the (μ, vec(triu(U))) layout of `pack_params`.
# ----------------------------------------------------------------------

function grad_explicit(n::Int, μ::Vector{Float64}, Σ::Matrix{Float64},
                       a::Vector{Float64}, b::Vector{Float64},
                       μ̂::Vector{Float64}, Σ̂::Matrix{Float64})
    set_kr_base_backend!(:mvnormalcdf)
    d = TruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4)
    g_μ, g_U = TruncatedDistributions.grad_true_loss(d, μ̂, Σ̂)
    return pack_params(g_μ, g_U)
end

function grad_forwarddiff(n::Int, θ::Vector{Float64},
                          a::Vector{Float64}, b::Vector{Float64},
                          μ̂::Vector{Float64}, Σ̂::Matrix{Float64})
    return ForwardDiff.gradient(θ -> loss(θ, n, a, b, μ̂, Σ̂), θ)
end

function grad_finitediff(n::Int, θ::Vector{Float64},
                         a::Vector{Float64}, b::Vector{Float64},
                         μ̂::Vector{Float64}, Σ̂::Matrix{Float64};
                         h::Float64 = 1e-5)
    L0 = loss(θ, n, a, b, μ̂, Σ̂)
    g = similar(θ)
    @inbounds for i in eachindex(θ)
        θp = copy(θ)
        θp[i] += h
        L1 = loss(θp, n, a, b, μ̂, Σ̂)
        g[i] = (L1 - L0) / h
    end
    return g
end

# ----------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------

struct GradConfig
    name::String
    n::Int
    descr::String
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
end

function bundled_grad_config(label, n, idx, descr)
    ex = TruncatedDistributions.get_example(n = n, index = idx)
    return GradConfig(label, n, descr,
                      Vector{Float64}(ex.μ), Matrix{Float64}(ex.Σ),
                      Vector{Float64}(ex.a), Vector{Float64}(ex.b))
end

# Univariate (n=1) examples: μ scalar, Σ 1×1.
function univariate_config(label, descr, μ_val, σ_val, a_val, b_val)
    return GradConfig(label, 1, descr,
                      [Float64(μ_val)], reshape([Float64(σ_val)^2], 1, 1),
                      [Float64(a_val)], [Float64(b_val)])
end

const CONFIGS = [
    # n = 1
    univariate_config("1A", "n=1: μ=0, σ=1, A=[-1, 1]",
                      0.0, 1.0, -1.0, 1.0),
    univariate_config("1B", "n=1: μ=0.5, σ=1.5, A=[-1, 2]",
                      0.5, 1.5, -1.0, 2.0),
    # n = 2: two bundled examples
    bundled_grad_config("2A", 2, 1, "bundled n=2 ex 1"),
    bundled_grad_config("2B", 2, 4, "bundled n=2 ex 4"),
    # n = 3: bundled
    bundled_grad_config("3A", 3, 1, "bundled n=3 ex 1"),
]

# ----------------------------------------------------------------------
# For each config: build the θ at which we time each gradient method,
# compute the gradients, time them, compare.
# ----------------------------------------------------------------------

# Construct a θ slightly perturbed from the true (μ, Σ) so the
# gradient is non-trivial but stays in the local basin of the true
# optimum (where μ̂, Σ̂ are the moments at the true parameters).
function eval_point(cfg::GradConfig; pert::Float64 = 0.05)
    n = cfg.n
    # μ̂, Σ̂ at the true (μ, Σ) — gradient at θ_true ≈ 0
    set_kr_base_backend!(:mvnormalcdf)
    d_true = TruncatedMvNormal(cfg.μ, cfg.Σ, cfg.a, cfg.b;
                               max_moment_levels = 4)
    μ̂ = collect(mean(d_true))
    Σ̂ = collect(cov(d_true))

    # Perturb μ and Σ by a small fraction.
    μ_pert = cfg.μ .+ pert .* sign.(randn(n))
    Σ_pert = cfg.Σ .* (1 + pert)
    # Ensure PD: identity-shift if needed.
    Σ_pert = (Σ_pert + Σ_pert') / 2 + pert * I
    U_pert = cholesky(Symmetric(inv(Σ_pert))).U
    θ_pert = pack_params(μ_pert, U_pert)
    return θ_pert, μ̂, Σ̂, μ_pert, Σ_pert
end

# ----------------------------------------------------------------------
# Formatters
# ----------------------------------------------------------------------

fmt_time(t) = t < 1e-3 ? @sprintf("%.0f\\,\\textmu s", t * 1e6) :
              t < 1.0  ? @sprintf("%.1f\\,ms",        t * 1e3) :
                         @sprintf("%.2f\\,s",         t)

fmt_err(e) = isnan(e) ? "--" :
             e < 1e-12 ? @sprintf("%.1g", e) :
                         @sprintf("%.1e", e)

# ----------------------------------------------------------------------
# Run + report
# ----------------------------------------------------------------------

struct GradResult
    name::String
    n::Int
    P::Int
    t_explicit::Float64
    t_fd::Float64
    t_ad::Float64
    err_ad::Float64
    err_fd::Float64
end

function run_grad(cfg::GradConfig; bench_samples = 3, bench_seconds = 30)
    Random.seed!(0)
    θ_pert, μ̂, Σ̂, μ_pert, Σ_pert = eval_point(cfg)
    n = cfg.n
    P = length(θ_pert)
    @info "config $(cfg.name) (n = $n, P = $P parameters): $(cfg.descr)"

    @info "    explicit (closed-form)..."
    g_exp = grad_explicit(n, μ_pert, Σ_pert, cfg.a, cfg.b, μ̂, Σ̂)
    t_exp = @belapsed grad_explicit($n, $μ_pert, $Σ_pert, $cfg.a, $cfg.b, $μ̂, $Σ̂) samples=bench_samples seconds=bench_seconds

    @info "    forward-mode AD..."
    g_ad = grad_forwarddiff(n, θ_pert, cfg.a, cfg.b, μ̂, Σ̂)
    t_ad = @belapsed grad_forwarddiff($n, $θ_pert, $cfg.a, $cfg.b, $μ̂, $Σ̂) samples=bench_samples seconds=bench_seconds

    @info "    finite differences..."
    g_fd = grad_finitediff(n, θ_pert, cfg.a, cfg.b, μ̂, Σ̂)
    t_fd = @belapsed grad_finitediff($n, $θ_pert, $cfg.a, $cfg.b, $μ̂, $Σ̂) samples=bench_samples seconds=bench_seconds

    # Relative errors versus the closed-form gradient.
    g_ref_norm = max(norm(g_exp), 1e-15)
    err_ad = norm(g_ad .- g_exp) / g_ref_norm
    err_fd = norm(g_fd .- g_exp) / g_ref_norm

    r = GradResult(cfg.name, n, P, t_exp, t_fd, t_ad, err_ad, err_fd)
    @info @sprintf("    DONE  explicit %s   AD %s (err %.1e)   FD %s (err %.1e)",
                    fmt_time(t_exp), fmt_time(t_ad), err_ad,
                    fmt_time(t_fd), err_fd)
    return r
end

function main()
    println("\n", "="^80)
    println("Gradient benchmark for §4 of the paper")
    println("="^80)

    results = GradResult[]
    for cfg in CONFIGS
        push!(results, run_grad(cfg))
    end

    println()
    @printf("%-4s | %3s | %3s | %-13s | %-13s | %-13s | %s\n",
            "Cfg", "n", "P", "explicit", "ForwardDiff", "FD (h=1e-5)", "AD err / FD err")
    println("-"^90)
    for r in results
        @printf("%-4s | %3d | %3d | %-13s | %-13s | %-13s | %.1e / %.1e\n",
                r.name, r.n, r.P,
                fmt_time(r.t_explicit), fmt_time(r.t_ad), fmt_time(r.t_fd),
                r.err_ad, r.err_fd)
    end

    println()
    println("LaTeX rows (paste into tab:grad-bench):")
    for r in results
        @printf("%s & %d & %d & %s & %s & %s & %s & %s \\\\\n",
                r.name, r.n, r.P,
                fmt_time(r.t_explicit),
                fmt_time(r.t_ad),
                fmt_time(r.t_fd),
                fmt_err(r.err_ad),
                fmt_err(r.err_fd))
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
