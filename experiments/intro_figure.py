#!/usr/bin/env python3
"""Figure for the paper's introduction: the worked n = 3 example.

Draws samples of the *fitted* untruncated normal N(mu*, Sigma*); the
draws landing inside the truncation box A are coloured by the density
(the point cloud itself carries the density information), the rejected
draws outside A are greyed out, and A is drawn as a dark wireframe
strictly inside the axes. Pale projections of the truncated law's
bivariate marginals sit on the far axis panes.

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

# Axis limits: the box sits strictly inside the axes.
LO = np.array([-2.6, -3.4, -3.0])
HI = np.array([3.6, 2.6, 2.4])

N_TOTAL = 3100  # ~57% land inside A, i.e. ~1800 accepted draws


def gauss_pdf(x, mu, sig_inv, log_norm):
    d = x - mu
    q = np.einsum("...i,ij,...j->...", d, sig_inv, d)
    return np.exp(log_norm - 0.5 * q)


SIG_INV = np.linalg.inv(SIGMA)
LOG_NORM = -0.5 * (3 * np.log(2 * np.pi) + np.log(np.linalg.det(SIGMA)))


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


def box_wireframe(ax):
    corners = [(xx, yy, zz) for xx in (A[0], B[0]) for yy in (A[1], B[1])
               for zz in (A[2], B[2])]
    for p in corners:
        for q in corners:
            diff = [abs(p[m] - q[m]) > 0 for m in range(3)]
            if sum(diff) == 1 and p < q:
                ax.plot(*zip(p, q), color="0.15", lw=1.3, zorder=6)


def main(outdir="."):
    L = np.linalg.cholesky(SIGMA)
    x = MU + rng.standard_normal((N_TOTAL, 3)) @ L.T
    # Drop the handful of draws beyond the axis limits.
    x = x[np.all((x >= LO) & (x <= HI), axis=1)]
    inside = np.all((x >= A) & (x <= B), axis=1)
    pdf_in = gauss_pdf(x[inside], MU, SIG_INV, LOG_NORM)

    fig = plt.figure(figsize=(7.0, 5.7))
    ax = fig.add_subplot(projection="3d")

    # Pale marginal projections of the *truncated* law on the far axis
    # panes (deliberately understated; the point cloud is the subject).
    pale = plt.get_cmap("Blues")
    for pair, zdir, offset in (((0, 1), "z", LO[2]),
                               ((0, 2), "y", HI[1]),
                               ((1, 2), "x", LO[0])):
        gi, gj, dens = wall_marginal(pair)
        GI, GJ = np.meshgrid(gi, gj, indexing="ij")
        kw = dict(levels=11, cmap=pale, alpha=0.75, zorder=1,
                  vmax=1.1 * dens.max())
        (i, j) = pair
        # Dashed frame: the box footprint projected onto the pane.
        fi = [A[i], B[i], B[i], A[i], A[i]]
        fj = [A[j], A[j], B[j], B[j], A[j]]
        frame_kw = dict(color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=2)
        if zdir == "z":
            ax.contourf(GI, GJ, dens, zdir="z", offset=offset, **kw)
            ax.plot(fi, fj, np.full(5, offset), **frame_kw)
        elif zdir == "y":
            ax.contourf(GI, dens, GJ, zdir="y", offset=offset, **kw)
            ax.plot(fi, np.full(5, offset), fj, **frame_kw)
        else:
            ax.contourf(dens, GI, GJ, zdir="x", offset=offset, **kw)
            ax.plot(np.full(5, offset), fi, fj, **frame_kw)

    # Rejected draws: grey, receding.
    ax.scatter(x[~inside, 0], x[~inside, 1], x[~inside, 2],
               color="0.62", s=3.5, alpha=0.35, linewidths=0, zorder=4)

    # Accepted draws: coloured by density — the cloud carries the
    # density information. Clip the low end of the colormap so the
    # faintest dots stay visible against white.
    ylorrd = matplotlib.colors.ListedColormap(
        plt.get_cmap("YlOrRd")(np.linspace(0.3, 1.0, 256)))
    ax.scatter(x[inside, 0], x[inside, 1], x[inside, 2],
               c=pdf_in, cmap=ylorrd, s=4.2, alpha=0.85,
               linewidths=0, zorder=10)

    box_wireframe(ax)

    ax.set_xlim(LO[0], HI[0])
    ax.set_ylim(LO[1], HI[1])
    ax.set_zlim(LO[2], HI[2])
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_zticks([-1.5, 0, 1])
    ax.set_zticklabels(["-1.5", "0", "1"])
    ax.set_xlabel(r"$x_1$", labelpad=-2)
    ax.set_ylabel(r"$x_2$", labelpad=-2)
    ax.set_zlabel(r"$x_3$", labelpad=-6)
    ax.tick_params(pad=-2, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("0.85")
    ax.grid(False)
    ax.view_init(elev=17, azim=-63)
    ax.set_box_aspect((HI[0] - LO[0], HI[1] - LO[1], HI[2] - LO[2]))
    fig.tight_layout(pad=0.2)

    fig.savefig(f"{outdir}/intro_example.png", dpi=220,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{outdir}/intro_example.pdf",
                bbox_inches="tight", pad_inches=0.05)
    n_in = int(inside.sum())
    print(f"wrote {outdir}/intro_example.[pdf|png]  "
          f"({n_in} inside, {len(x) - n_in} outside)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
