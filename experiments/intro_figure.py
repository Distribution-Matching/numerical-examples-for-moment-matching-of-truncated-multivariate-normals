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
MU = np.array([0.5047, 0.0035, -0.3012])
SIGMA = np.array([
    [1.1041, 0.5490, 0.3334],
    [0.5490, 1.0147, 0.2082],
    [0.3334, 0.2082, 0.9985],
])
# Truncation region: two-sided box on x1, x2; a one-sided floor at
# -1.5 on x3 (upper bound +inf).
A = np.array([-1.0, -2.0, -1.5])
B = np.array([2.0, 1.0, np.inf])

# Axis limits: the box sits strictly inside the axes on the bounded
# faces; on x3 it is open above and runs to the top of the plot.
LO = np.array([-2.6, -3.4, -3.0])
HI = np.array([3.6, 2.6, 2.4])

# Finite upper bounds for plotting (BTOP) and for the x3 integration in
# the marginal that integrates x3 out (BINT: far into the tail).
BTOP = np.where(np.isfinite(B), B, HI)
BINT = np.where(np.isfinite(B), B, MU + 8.0 * np.sqrt(np.diag(SIGMA)))

N_TOTAL = 2800  # ~64% land inside A, i.e. ~1800 accepted draws


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
    # Grid axes are plotted up to their finite top (BTOP); the
    # integrated-out axis k is carried far into the tail (BINT).
    gi = np.linspace(A[i], BTOP[i], n_grid)
    gj = np.linspace(A[j], BTOP[j], n_grid)
    tq, wq = np.polynomial.legendre.leggauss(n_quad)
    tk = 0.5 * (A[k] + BINT[k]) + 0.5 * (BINT[k] - A[k]) * tq
    wk = 0.5 * (BINT[k] - A[k]) * wq
    GI, GJ, TK = np.meshgrid(gi, gj, tk, indexing="ij")
    pts = np.empty(GI.shape + (3,))
    pts[..., i] = GI
    pts[..., j] = GJ
    pts[..., k] = TK
    dens = (gauss_pdf(pts, MU, SIG_INV, LOG_NORM) * wk).sum(axis=-1)
    return gi, gj, dens


def box_wireframe(ax):
    # x3 is unbounded above: draw the four vertical walls up to the top of
    # the plot (BTOP[2]) and the floor, but omit the top face so the box
    # reads as open — it continues upward. Solid edges up to the last
    # bounded level; a short dashed stub past it signals the open top.
    ztop = BTOP[2]
    zmark = ztop - 0.28 * (ztop - A[2])   # where the solid edge stops
    for xx in (A[0], B[0]):
        for yy in (A[1], B[1]):
            ax.plot([xx, xx], [yy, yy], [A[2], zmark],
                    color="0.15", lw=1.3, zorder=6)
            ax.plot([xx, xx], [yy, yy], [zmark, ztop],
                    color="0.15", lw=1.3, ls=(0, (2, 2)), zorder=6)
    # Floor face (z = A[2]): four edges.
    floor = [(A[0], A[1]), (B[0], A[1]), (B[0], B[1]), (A[0], B[1]),
             (A[0], A[1])]
    for (p, q) in zip(floor[:-1], floor[1:]):
        ax.plot([p[0], q[0]], [p[1], q[1]], [A[2], A[2]],
                color="0.15", lw=1.3, zorder=6)


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
        (i, j) = pair
        # x3 is unbounded above (its axis is index 2 on these panes): the
        # marginal should read as open at the top, not capped.
        open_top = 2 in pair
        if open_top:
            # Floor the lowest contour level so the faint tail does not
            # fill up to a hard horizontal edge at the plot top; the fill
            # closes off and fades out well below it.
            vmax = 1.1 * dens.max()
            kw = dict(levels=np.linspace(0.09 * dens.max(), vmax, 11),
                      cmap=pale, alpha=0.75, zorder=1, vmax=vmax)
        else:
            kw = dict(levels=11, cmap=pale, alpha=0.75, zorder=1,
                      vmax=1.1 * dens.max())
        # Dashed frame: the box footprint projected onto the pane. On the
        # x3 panes the top edge is dropped (open above); the bounded x1-x2
        # pane keeps a closed rectangle.
        if open_top:
            fi = [A[i], A[i], BTOP[i], BTOP[i]]
            fj = [BTOP[j], A[j], A[j], BTOP[j]]
        else:
            fi = [A[i], BTOP[i], BTOP[i], A[i], A[i]]
            fj = [A[j], A[j], BTOP[j], BTOP[j], A[j]]
        frame_kw = dict(color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=2)
        if zdir == "z":
            ax.contourf(GI, GJ, dens, zdir="z", offset=offset, **kw)
            ax.plot(fi, fj, np.full(len(fi), offset), **frame_kw)
        elif zdir == "y":
            ax.contourf(GI, dens, GJ, zdir="y", offset=offset, **kw)
            ax.plot(fi, np.full(len(fi), offset), fj, **frame_kw)
        else:
            ax.contourf(dens, GI, GJ, zdir="x", offset=offset, **kw)
            ax.plot(np.full(len(fi), offset), fi, fj, **frame_kw)

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
    ax.set_zticks([-1.5, 0, 2])
    ax.set_zticklabels(["-1.5", "0", "2"])
    ax.set_xlabel(r"$x_1$", labelpad=-2)
    ax.set_ylabel(r"$x_2$", labelpad=-2)
    ax.set_zlabel(r"$x_3$", labelpad=-6)
    ax.tick_params(pad=-2, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        # No pane outlines: the faint grey box edges (in particular the top
        # face) would suggest a ceiling, but x3 is unbounded above.
        axis.pane.set_edgecolor("none")
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
