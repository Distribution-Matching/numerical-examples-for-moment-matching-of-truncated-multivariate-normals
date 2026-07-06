#!/usr/bin/env python3
"""Figure for Section 5: a hybrid-BCD run as a loss curve annotated with
the sub-block visited at each iteration.

Reads the trace produced by `bcd_trace.jl` (iteration, sub-block S, block
size k, accepted flag, full loss L after the iteration) and draws L
against iteration on a log axis. Each iteration is a marker on the curve
labelled by its sub-block $S$: filled = accepted (the write-back lowered
the full loss and was kept), open = rejected (loss unchanged). Under
full-loss acceptance the curve is monotone, stepping down only at the
accepts.

Usage:  python3 experiments/bcd_trace_plot.py <data.txt> [outdir]
The data file has a `# ...` header then rows `iter S k accept L`.
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_trace(path):
    rows = []
    meta = ""
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            meta = line
            continue
        it, S, k, acc, L = line.split()
        rows.append((int(it), S, int(k), int(acc) == 1, float(L)))
    return rows, meta


def main(path, outdir="."):
    rows, _ = read_trace(path)
    it = np.array([r[0] for r in rows])
    L = np.array([r[4] for r in rows])
    acc = np.array([r[3] for r in rows])
    Slab = [r[1] for r in rows]

    acc_col = "#b0421f"   # accepted
    rej_col = "0.55"      # rejected

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    # The loss curve.
    ax.plot(it, L, "-", color="0.4", lw=1.1, zorder=1)
    ax.scatter(it[acc],  L[acc],  s=42, color=acc_col, zorder=4,
               label="accepted")
    ax.scatter(it[~acc], L[~acc], s=34, facecolors="white",
               edgecolors=rej_col, linewidths=1.2, zorder=3,
               label="rejected")

    ax.set_yscale("log")
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel(r"full loss $L$")
    ax.set_xlim(it.min() - 0.6, it.max() + 0.6)
    ax.set_xticks([t for t in it if t % 2 == 0 or t == it[-1]])
    lo, hi = L.min(), L.max()
    ax.set_ylim(lo * 0.5, hi * 2.2)

    # Annotate each iteration with its sub-block S, alternating above /
    # below the marker to reduce collisions.
    for i, (x, y, s, a) in enumerate(zip(it, L, Slab, acc)):
        if s == "-":            # the start point (iteration 0)
            continue
        above = (i % 2 == 0)
        ax.annotate(
            "$\\{" + s.replace(",", ",") + "\\}$",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9 if above else -14),
            ha="center", va="bottom" if above else "top",
            fontsize=7.5,
            color=acc_col if a else rej_col,
        )

    ax.legend(loc="upper right", frameon=False, fontsize=9,
              handletextpad=0.4)
    ax.grid(True, which="major", axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)

    fig.savefig(f"{outdir}/bcd_trace.png", dpi=220,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{outdir}/bcd_trace.pdf",
                bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {outdir}/bcd_trace.[pdf|png]")


if __name__ == "__main__":
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    main(path, outdir)
