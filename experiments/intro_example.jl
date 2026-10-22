#=
intro_example.jl

The paper introduction's worked n = 3 example: the forward truncated
moments of (μ, Σ) on A = [-1, 2] × [-2, 1] × [-1.5, ∞), and the inverse
fits (normal and Student-t₅) to the rounded targets (μ̃, Σ̃), with the
terminal loss and the box probability of each fit. The fitted normal
parameters feed intro_figure.py.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/intro_example.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)

const SEED = 20260706

μ = [0.5, 0.0, -0.3]
Σ = [1.0 0.5 0.3; 0.5 1.0 0.2; 0.3 0.2 1.0]
a = [-1.0, -2.0, -1.5]
b = [2.0, 1.0, Inf]

μ̃ = [0.47, -0.19, -0.12]
Σ̃ = [0.54 0.16 0.11; 0.16 0.49 0.05; 0.11 0.05 0.65]

showvec(v) = join((@sprintf("%.3f", x) for x in v), ", ")
showmat(M) = join((showvec(M[i, :]) for i in 1:size(M, 1)), "; ")

d = TruncatedMvNormal(μ, Σ, a, b)
println("forward:  P(X∈A) = ", @sprintf("%.3f", tp(d)))
println("          μ_A = [", showvec(mean(d)), "]")
println("          Σ_A = [", showmat(Matrix(cov(d))), "]")

kw = (; gamma = 1.0, ftarget = 1e-12, iterations = 300, time_limit = 300.0, seed = SEED)

μN, ΣN, infoN = fit_mvnormal(μ̃, Σ̃, a, b; kw...)
dN = TruncatedMvNormal(μN, Matrix(ΣN), a, b)
println("\nnormal fit:  loss = ", @sprintf("%.1e", infoN.loss),
        ",  P(X∈A) = ", @sprintf("%.3f", tp(dN)))
println("  μ* = [", showvec(μN), "]")
println("  Σ* = [", showmat(Matrix(ΣN)), "]")
println("  truncated moments: μ_A = [", showvec(mean(dN)), "]")
println("                     Σ_A = [", showmat(Matrix(cov(dN))), "]")

ν = 5
μT, ΣT, infoT = fit_mvt(ν, μ̃, Σ̃, a, b; kw...)
dT = TruncatedMvT(ν, μT, Matrix(ΣT), a, b)
println("\nt_5 fit:  loss = ", @sprintf("%.1e", infoT.loss),
        ",  P(X∈A) = ", @sprintf("%.3f", tp(dT)))
println("  μ*_t = [", showvec(μT), "]")
println("  Σ*_t = [", showmat(Matrix(ΣT)), "]")
println("  truncated moments: μ_A = [", showvec(mean(dT)), "]")
println("                     Σ_A = [", showmat(Matrix(cov(dT))), "]")
