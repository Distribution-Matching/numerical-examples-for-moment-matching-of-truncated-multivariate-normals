#=
t_gradient_table.jl

Companion to the paper's Student-t gradient table, the counterpart of
gradient_table.jl for the truncated t_ν (ν = 5). Wall-clock and accuracy
of the gradient of the γ-weighted moment-matching loss (γ = 1) computed
three ways:

  (a) closed form (Corollary t-gradient), vector_fg_true_loss_t, on the
      Genz–Bretz QMC base case (cubature at n ≤ 2, as in the package);
  (b) forward-mode AD (ForwardDiff.jl) through a type-generic loss built
      by direct adaptive cubature of the t density (t_generic_loss.jl,
      rtol = 1e-8);
  (c) one-sided finite differences (h = 1e-5) on the same generic loss.

As in gradient_table.jl, targets are the truncated moments at the true
(μ, Σ) and the gradient is evaluated at a small seeded perturbation; AD
and FD run at n ≤ 3 only. For reference, the last timing column is the
closed-form gradient of the *normal* loss at the same point.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_gradient_table.jl
=#

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "t_generic_loss.jl"))
using ForwardDiff
using BenchmarkTools
using Printf
using Random

BLAS.set_num_threads(1)

const NU = 5.0
const SEED = 20261015
const AD_FD_MAX_N = 3

struct Cfg
    name::String
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
end
bundled(name, n, idx) = (ex = get_example(n = n, index = idx);
    Cfg(name, collect(Float64, ex.μ), Matrix{Float64}(ex.Σ),
        collect(Float64, ex.a), collect(Float64, ex.b)))
univ(name, μ, σ, a, b) = Cfg(name, [μ], fill(σ^2, 1, 1), [a], [b])

const CONFIGS = [
    univ("1.1", 0.0, 1.0, -1.0, 1.0),
    univ("1.2", 0.5, 1.5, -1.0, 2.0),
    bundled("2.1", 2, 1), bundled("2.4", 2, 4),
    bundled("3.1", 3, 1),
    bundled("4.1", 4, 1), bundled("5.1", 5, 1), bundled("6.1", 6, 1),
]

function eval_point(c::Cfg; pert = 0.05)
    n = length(c.μ)
    set_kr_base_backend!(:hcubature)
    d = TruncatedMvT(NU, c.μ, c.Σ, c.a, c.b; max_moment_levels = 2)
    μ̂ = mean(d); Σ̂ = Matrix(cov(d))
    rng = MersenneTwister(SEED)
    μp = c.μ .+ pert .* sign.(randn(rng, n))
    Σp = (c.Σ .* (1 + pert)); Σp = (Σp + Σp') / 2 + pert * I
    return make_param_vec_from_μ_Σ(μp, Σp), μ̂, Σ̂
end

function grad_closed(p, c, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf)
    set_kr_base_rng!(MersenneTwister(SEED))
    G = similar(p)
    vector_fg_true_loss_t(nothing, G, p, NU, c.a, c.b, μ̂, Σ̂)
    return G
end
function grad_normal(p, c, μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf)
    set_kr_base_rng!(MersenneTwister(SEED))
    G = similar(p)
    vector_fg_true_loss(nothing, G, p, c.a, c.b, μ̂, Σ̂)
    return G
end
gen_loss(p, c, μ̂, Σ̂) = vector_moment_loss_t_hcub(p, NU, c.a, c.b, μ̂, Σ̂; rtol = 1e-8)
grad_ad(p, c, μ̂, Σ̂) = ForwardDiff.gradient(q -> gen_loss(q, c, μ̂, Σ̂), p)
function grad_fd(p, c, μ̂, Σ̂; h = 1e-5)
    L0 = gen_loss(p, c, μ̂, Σ̂)
    return [(q = copy(p); q[i] += h; (gen_loss(q, c, μ̂, Σ̂) - L0) / h) for i in eachindex(p)]
end

fmt_time(t) = isnan(t) ? "--" :
              t < 1e-3 ? @sprintf("\$%.0f\$\\,\\textmu s", t * 1e6) :
              t < 1.0  ? @sprintf("\$%.1f\$\\,ms", t * 1e3) :
                         @sprintf("\$%.2f\$\\,s", t)
fmt_err(e) = isnan(e) ? "--" : replace(@sprintf("\$%.1e\$", e), r"e-0?(\d+)" => s"\\mathrm{e}\\text{-}\1")

function main()
    set_loss_gamma!(1.0)
    latex = String[]
    @printf("%-4s %2s %3s %12s %12s %12s %10s %10s %12s\n", "Cfg", "n", "P", "closed t",
            "ForwardDiff", "FD", "AD err", "FD err", "closed N")
    for c in CONFIGS
        n = length(c.μ)
        p, μ̂, Σ̂ = eval_point(c)
        g = grad_closed(p, c, μ̂, Σ̂)
        t_c = @belapsed grad_closed($p, $c, $μ̂, $Σ̂) samples = 3 seconds = 30
        t_n = @belapsed grad_normal($p, $c, $μ̂, $Σ̂) samples = 3 seconds = 30
        t_ad = t_fd = e_ad = e_fd = NaN
        if n <= AD_FD_MAX_N
            gad = grad_ad(p, c, μ̂, Σ̂)
            t_ad = @belapsed grad_ad($p, $c, $μ̂, $Σ̂) samples = 3 seconds = 30
            gfd = grad_fd(p, c, μ̂, Σ̂)
            t_fd = @belapsed grad_fd($p, $c, $μ̂, $Σ̂) samples = 3 seconds = 30
            e_ad = norm(gad - g) / norm(g)
            e_fd = norm(gfd - g) / norm(g)
        end
        @printf("%-4s %2d %3d %12.2e %12.2e %12.2e %10.1e %10.1e %12.2e\n", c.name, n,
                length(p), t_c, t_ad, t_fd, e_ad, e_fd, t_n)
        flush(stdout)
        push!(latex, @sprintf("%s & %d & %2d & %s & %s & %s & %s & %s & %s \\\\", c.name, n,
                              length(p), fmt_time(t_c), fmt_time(t_ad), fmt_time(t_fd),
                              fmt_err(e_ad), fmt_err(e_fd), fmt_time(t_n)))
    end
    println("\nLaTeX rows:")
    foreach(println, latex)
end

main()
