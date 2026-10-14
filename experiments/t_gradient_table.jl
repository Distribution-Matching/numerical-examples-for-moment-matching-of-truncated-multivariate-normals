#=
t_gradient_table.jl

Companion to the paper's Student-t cost table. For the first bundled
example at each n = 2, …, 5, one evaluation of the loss and closed-form
gradient for the truncated t_ν (ν = 5) against the same for the truncated
normal, both on the Genz–Bretz QMC base case; and, at n ≤ 4, the accuracy
of the QMC t gradient against the same closed-form gradient on an
adaptive-cubature base case (relative norm error). Timings are medians
of repeated post-JIT evaluations.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_gradient_table.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using LinearAlgebra
using Printf
using Random
using Statistics

BLAS.set_num_threads(1)

const NU = 5.0
const SEED = 20261015

function med_time(f; reps)
    f()
    return median([@elapsed(f()) for _ in 1:reps])
end

function main()
    set_loss_gamma!(1.0)
    println(@sprintf("%-3s %4s %6s %12s %12s %8s %12s", "n", "P", "M_t",
                     "t-grad (s)", "N-grad (s)", "ratio", "t rel.err"))
    latex = String[]
    for n in 2:5
        ex = get_example(n = n, index = 1)
        μ = collect(Float64, ex.μ); Σ = Matrix{Float64}(ex.Σ)
        a = collect(Float64, ex.a); b = collect(Float64, ex.b)
        μ̂ = μ .+ 0.05; Σ̂ = 0.8 .* Σ               # gradient is non-trivial here
        p = make_param_vec_from_μ_Σ(μ, Σ)
        P = length(p)
        G = similar(p)
        reps = n <= 3 ? 7 : 3
        set_kr_base_backend!(:mvnormalcdf)
        set_kr_base_rng!(MersenneTwister(SEED))
        tt = med_time(() -> vector_fg_true_loss_t(1, G, p, NU, a, b, μ̂, Σ̂); reps)
        Gq = copy(G)
        tn = med_time(() -> vector_fg_true_loss(1, G, p, a, b, μ̂, Σ̂); reps)
        err = NaN
        if n <= 4
            set_kr_base_backend!(:hcubature)
            vector_fg_true_loss_t(1, G, p, NU, a, b, μ̂, Σ̂)
            err = norm(Gq - G) / norm(G)
        end
        Mt = binomial(n + 2, 2) + binomial(n + 4, 4)
        println(@sprintf("%-3d %4d %6d %12.4f %12.4f %8.1f %12.1e", n, P, Mt, tt, tn,
                         tt / tn, err))
        flush(stdout)
        push!(latex, @sprintf("%d & %d & %d & %.3g & %.3g & %.0f & %s \\\\", n, P, Mt,
                              tt, tn, tt / tn,
                              isnan(err) ? "--" : @sprintf("%.1e", err)))
    end
    println("\nLaTeX rows:")
    foreach(println, latex)
end

main()
