#=
example_2_3_longrun.jl

Companion to §6 of the paper. Runs L-BFGS on example 2.3 well past the
benchmark's 50-iteration cap, to back the statements that its rounded
target appears unattainable: the loss levels off while the parameters
keep growing. Same targets, naive start and loss (γ = 1) as fit_table.jl
and t_fit_table.jl, but with the deterministic cubature base case, since
quasi-Monte-Carlo noise ends the normal run early (the last lines show
this for four seeds). Each restart resumes L-BFGS from the previous end
point with a fresh curvature memory.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/example_2_3_longrun.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using Optim
using Optim.NLSolversBase: only_fg!
using LinearAlgebra
using Printf
using Random

BLAS.set_num_threads(1)

const SEED = 20260706

ex = get_example(n = 2, index = 3)
a = collect(Float64, ex.a)
b = collect(Float64, ex.b)
set_loss_gamma!(1.0)

function rounded_targets(ν)
    set_kr_base_rng!(MersenneTwister(SEED))
    if ν === nothing
        set_kr_base_backend!(:mvnormalcdf)
        d = TruncatedMvNormal(collect(Float64, ex.μ), Matrix{Float64}(ex.Σ), a, b)
    else
        set_kr_base_backend!(:hcubature)
        d = TruncatedMvT(ν, collect(Float64, ex.μ), Matrix{Float64}(ex.Σ), a, b)
    end
    return round.(collect(mean(d)); digits = 1), round.(Matrix(cov(d)); digits = 1)
end

function report(label, r)
    μ, Σ = make_μ_Σ_from_param_vec(r.minimizer)
    @printf("%-30s iterations = %5d   L = %.3e   μ = %s   diag(Σ) = %s\n",
            label, r.iterations, r.minimum,
            round.(μ; sigdigits = 3), round.(diag(Σ); sigdigits = 3))
    flush(stdout)
end

println("example 2.3: μ = ", collect(ex.μ), ", diag(Σ) = ", diag(Matrix(ex.Σ)))

opts() = Optim.Options(iterations = 3000, time_limit = 100.0, g_abstol = 0.0,
                       callback = s -> s.f_x < 1e-10)

let (μ̂, Σ̂) = rounded_targets(nothing)
    println("normal targets: μ̂ = ", μ̂, ", Σ̂ = ", Σ̂)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Σ̂)
    set_kr_base_backend!(:hcubature)
    p = make_param_vec_from_μ_Σ(μ̂, Σ̂)
    for k in 0:3
        r = optimize(only_fg!(fg!), p, LBFGS(), opts())
        report("normal cubature, restart $k", r)
        p = r.minimizer
    end
    set_kr_base_backend!(:mvnormalcdf)
    for seed in (1, 2, 3, SEED)
        set_kr_base_rng!(MersenneTwister(seed))
        r = optimize(only_fg!(fg!), make_param_vec_from_μ_Σ(μ̂, Σ̂), LBFGS(), opts())
        report("normal QMC, seed $seed", r)
    end
end

for ν in (4.0, 10.0)
    μ̂, Σ̂ = rounded_targets(ν)
    println("t targets, ν = $ν: μ̂ = ", μ̂, ", Σ̂ = ", Σ̂)
    set_kr_base_backend!(:hcubature)
    fg!(F, G, p) = vector_fg_true_loss_t(F, G, p, ν, a, b, μ̂, Σ̂)
    p = make_param_vec_from_μ_Σ(μ̂, ((ν - 2) / ν) .* Σ̂)
    for k in 0:1
        r = optimize(only_fg!(fg!), p, LBFGS(), opts())
        report("t ν=$(Int(ν)) cubature, restart $k", r)
        p = r.minimizer
    end
end
