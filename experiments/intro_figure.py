#!/usr/bin/env python3
"""Figure for the paper's introduction: the worked n = 3 example.

Draws samples of the *fitted* box-truncated normal N(mu*, Sigma*) | A
(whose truncated moments equal the desired (mu_hat, Sigma_hat)) as a 3-D
point cloud, with the exact bivariate marginal densities of the truncated
law painted on three walls of the box.

(mu*, Sigma*) is the output of TruncatedDistributions.fit_mvnormal for
the targets quoted in the introduction (loss ~ 2e-11); see the paper.

Usage:  python3 experiments/intro_figure.py [outdir]
Writes intro_example.pdf and intro_example.png to outdir (default '.').
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(7)

# Fitted parameters (see paper introduction).
MU = np.array([0.4927, 0.0018, -0.3162])
SIGMA = np.array([
    [1.0005, 0.4855, 0.3190],
    [0.4855, 0.9797, 0.1924],
    [0.3190, 0.1924, 1.0481],
])
A = np.array([-1.0, -2.0, -1.5])
B = np.array([2.0, 1.0, 1.0])

N_DOTS = 1400


def gauss_pdf(x, mu, sig_inv, log_norm):
    d = x - mu
    q = np.einsum("...i,ij,...j->...", d, sig_inv, d)
    return np.exp(log_norm - 0.5 * q)


SIG_INV = np.linalg.inv(SIGMA)
LOG_NORM = -0.5 * (3 * np.log(2 * np.pi) + np.log(np.linalg.det(SIGMA)))


def sample_truncated(n):
    out = []
    L = np.linalg.cholesky(SIGMA)
    while sum(len(o) for o in out) < n:
        z = rng.standard_normal((4 * n, 3))
        x = MU + z @ L.T
        keep = np.all((x >= A) & (x <= B), axis=1)
        out.append(x[keep])
    return np.concatenate(out)[:n]


def wall_marginal(ax_pair, n_grid=90, n_quad=48):
    """Exact bivariate marginal of the truncated law on axes `ax_pair`,
    integrating the trivariate Gaussian over the remaining axis with
    Gauss-Legendre quadrature. Returns (G1, G2, density)."""
    (i, j) = ax_pair
    k = ({0, 1, 2} - {i, j}).pop()
    gi = np.linspace(A[i], B[i], n_grid)
    gj = np.linspace(A[j], B[j], n_grid)
    tq, wq = np.polynomial.legendre.leggauss(n_quad)
    tk = 0.5 * (A[k] + B[k]) + 0.5 * (B[k] - A[k]) * tq
    wk = 0.5 * (B[k] - A[k]) * wq
    GI, GJ, TK = np.meshgrid(gi, gj, tk, indexing="ij")
    pts = np.empty(GI.shape + (3,))
    pts[..., i] = GI
    pts[..., j] = GJ
    pts[..., k] = TK
    dens = (gauss_pdf(pts, MU, SIG_INV, LOG_NORM) * wk).sum(axis=-1)
    return gi, gj, dens


def main(outdir="."):
    x = sample_truncated(N_DOTS)

    fig = plt.figure(figsize=(6.6, 5.4))
    ax = fig.add_subplot(projection="3d")

    cmap_wall = plt.get_cmap("Blues")
    dot_color = "#b0421f"

    # Walls: (x1,x2) painted at x3 = A[2]; (x1,x3) at x2 = B[1];
    # (x2,x3) at x1 = A[0].
    g1, g2, d12 = wall_marginal((0, 1))
    G1, G2 = np.meshgrid(g1, g2, indexing="ij")
    ax.contourf(G1, G2, d12, zdir="z", offset=A[2], levels=14,
                cmap=cmap_wall, alpha=0.85, zorder=1)

    g1, g3, d13 = wall_marginal((0, 2))
    G1, G3 = np.meshgrid(g1, g3, indexing="ij")
    ax.contourf(G1, d13, G3, zdir="y", offset=B[1], levels=14,
                cmap=cmap_wall, alpha=0.85, zorder=1)

    g2, g3, d23 = wall_marginal((1, 2))
    G2, G3 = np.meshgrid(g2, g3, indexing="ij")
    ax.contourf(d23, G2, G3, zdir="x", offset=A[0], levels=14,
                cmap=cmap_wall, alpha=0.85, zorder=1)

    # Dots.
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=dot_color,
               s=7.5, alpha=0.55, linewidths=0, zorder=10)

    # Box wireframe.
    corners = [(xx, yy, zz) for xx in (A[0], B[0]) for yy in (A[1], B[1])
               for zz in (A[2], B[2])]
    for p in corners:
        for q in corners:
            diff = [abs(p[m] - q[m]) > 0 for m in range(3)]
            if sum(diff) == 1 and p < q:
                ax.plot(*zip(p, q), color="0.35", lw=0.8, zorder=5)

    ax.set_xlim(A[0], B[0])
    ax.set_ylim(A[1], B[1])
    ax.set_zlim(A[2], B[2])
    ax.set_xticks([0, 1])
    ax.set_yticks([-1, 0])
    ax.set_zticks([-1, 0])
    ax.set_xlabel(r"$x_1$", labelpad=-2)
    ax.set_ylabel(r"$x_2$", labelpad=-2)
    ax.set_zlabel(r"$x_3$", labelpad=-6)
    ax.tick_params(pad=-2, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("0.85")
    ax.grid(False)
    ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect((B[0] - A[0], B[1] - A[1], B[2] - A[2]))
    fig.tight_layout(pad=0.2)

    fig.savefig(f"{outdir}/intro_example.png", dpi=220,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{outdir}/intro_example.pdf",
                bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {outdir}/intro_example.[pdf|png]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
