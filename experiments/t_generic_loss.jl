#=
t_generic_loss.jl

Type-generic moment-matching loss for the box-truncated multivariate
Student-t, by direct adaptive cubature of the t density. This is the
Student-t counterpart of the generic cubature loss used for the
finite-difference and ForwardDiff baselines of the Gaussian tables: it
accepts ForwardDiff.Dual parameters, so both baselines differentiate
through it. Included by t_gradient_table.jl and t_fit_table.jl.

The loss is the package's γ-weighted L = ½‖μA − μ̂‖² + γ·r(n)·½‖ΣA − Σ̂‖²_F
with r(n) = 2/(n+1), on the packed (μ, U) vector (UᵀU = Σ⁻¹).
=#

using TruncatedDistributions
using Distributions
using LinearAlgebra

function unpack_generic_t(p::AbstractVector{T}, n::Int) where {T}
    μ = p[1:n]
    U = zeros(T, n, n)
    k = n + 1
    for i in 1:n, j in i:n
        U[i, j] = p[k]
        k += 1
    end
    return μ, U
end

# log c_{n,ν}: the log normalising constant of the standard t_ν in n dims.
log_c_t(ν, n) = logpdf(MvTDist(Float64(ν), zeros(n), Matrix{Float64}(I, n, n)), zeros(n))

function vector_moment_loss_t_hcub(p::AbstractVector{T}, ν, a, b, μ̂, Σ̂;
                                   rtol = 1e-5, maxevals = 10_000_000) where {T}
    n = length(μ̂)
    μ, U = unpack_generic_t(p, n)
    logc = log_c_t(ν, n)
    logdetU = sum(log, abs.(diag(U)))
    function dens(x)
        z = U * (x .- μ)
        return exp(logc + logdetU - (n + ν) / 2 * log1p(dot(z, z) / ν))
    end
    m0 = hcubature_inf(dens, a, b; rtol = rtol, maxevals = maxevals)[1]
    m1 = [hcubature_inf(x -> x[i] * dens(x), a, b; rtol = rtol, maxevals = maxevals)[1]
          for i in 1:n]
    m2 = Matrix{T}(undef, n, n)
    for i in 1:n, j in i:n
        m2[i, j] = m2[j, i] = hcubature_inf(x -> x[i] * x[j] * dens(x), a, b;
                                            rtol = rtol, maxevals = maxevals)[1]
    end
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'
    w2 = get_loss_gamma() * 2 / (n + 1)
    return T(0.5) * (sum(abs2, μA .- μ̂) + w2 * sum(abs2, ΣA .- Σ̂))
end
