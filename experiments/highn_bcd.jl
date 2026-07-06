#=
highn_bcd.jl — probe hybrid BCD past the n = 10 panel.

Ground-truth moments at high n use the package's Monte-Carlo path
(`mc_moments` / `BasicBoxTruncatedMvNormal`), since the Kan–Robotti
recursion cannot even be built there. For each n we build a known
centred Gaussian with a tridiagonal correlation band on a box, take its
MC moments (rounded) as the target, fit with marginal-acceptance BCD,
and verify by MC on the fitted parameters.

Run:  OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/highn_bcd.jl
=#
using Pkg; Pkg.activate(@__DIR__)
using TruncatedDistributions, LinearAlgebra, Random, Printf, Statistics
BLAS.set_num_threads(1)
const SEED = 20260706
const NSAMP = 200_000

tridiag(n; ρ=0.3) = (M=Matrix(1.0I,n,n); for i in 1:n-1; M[i,i+1]=ρ; M[i+1,i]=ρ; end; M)

function probe(n; c=2.0)
    Σ = tridiag(n); a = fill(-c,n); b = fill(c,n)
    d = BasicBoxTruncatedMvNormal(zeros(n), Σ, a, b)
    tp, μt, Σt = mc_moments(d; n_samples=NSAMP, rng=MersenneTwister(SEED))
    μ̂ = round.(μt; digits=1); Σ̂ = round.(Σt; digits=1)
    t = @elapsed begin
        μf, Σf, info = fit_mvnormal(μ̂, Σ̂, a, b; method=:bcd, gamma=1.0,
                                    bcd_selection=:softmax, bcd_accept_by=:marginal,
                                    seed=SEED, iterations=40, verbose=false)
    end
    # verify by MC on the fitted parameters
    df = BasicBoxTruncatedMvNormal(collect(μf), Matrix(Σf), a, b)
    _, μv, Σv = mc_moments(df; n_samples=NSAMP, rng=MersenneTwister(SEED+9))
    dμ = norm(μv .- μ̂); dΣ = norm(Σv .- Σ̂)
    @printf("n=%2d  tp=%.2f  fit %5.1fs (%d it)  marg-L=%.1e  verify ‖Δμ‖=%.3f ‖ΔΣ‖_F=%.3f\n",
            n, tp, t, info.iterations, info.loss, dμ, dΣ)
end

set_moment_mc!(above=6, samples=NSAMP)
probe(8)   # warmup + sanity
for n in (12, 15, 20, 25)
    probe(n)
end
