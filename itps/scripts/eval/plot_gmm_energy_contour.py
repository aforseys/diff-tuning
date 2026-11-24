#!/usr/bin/env python
"""
2D contour map of a GMM EBM's energy at one denoising timestep, in the front figure's style
(notebooks/front_figure.ipynb): no axes, labels or colorbar, viridis_r stopped at 0.8 of its range
so the highest energies stay a mid blue. No cluster dots -- this is the bare landscape.

Reads the same .npz that `scripts/eval/eval.py --viz --viz-dir <dir>` writes
(<dir>/energy_surface_s{k}_t{ttt}_obs0.npz, keys x, y, energy) and normalizes it exactly as the 2D
and 3D panels do (plot_gmm_dmse.unit_normalize): min-max to [0,1] over the plotted area, per map.
Lower energy = the model considers that point more likely, so likely regions are the bright valleys.

With --arrows, a quiver of the denoising direction -dE/dx is drawn on top. Lengths are STANDARDIZED
WITHIN EACH IMAGE -- the longest arrow spans one arrow spacing and every other is proportional to
its |grad E| -- which is the convention in both front_figure.ipynb cell 5 and
envs/gmm/eval.viz_gradient_field. So arrow length is comparable within a figure but NOT across
figures or timesteps.

The gradient is a central difference of the saved energy grid rather than the model's exact
eps_hat (eval.py saves the gradient only as a .png). For an EBM those agree in principle --
eps_hat IS dE/dx -- and differencing the drawn grid guarantees the arrows sit perpendicular to the
drawn contours. It does smooth features narrower than a couple of grid cells (spacing ~0.1 here).
For the exact field, use envs/gmm/eval.viz_gradient_field, which calls the model.

Usage (from itps/, with the repo root on PYTHONPATH):
  python scripts/eval/plot_gmm_energy_contour.py \
      --dirs "Base (EBM)=outputs/energy/ebm_base" \
      --timesteps 10 --arrows --out-dir outputs/figures/energy_contour
"""

import argparse
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# Sibling scripts; python puts this file's directory on sys.path.
from plot_gmm_dmse import unit_normalize
from plot_gmm_energy_3d import find_map, save, truncated_cmap


def gradient_arrows(x, y, energy, lim, stride):
    """
    -grad E on every `stride`-th grid point inside `lim`, and its magnitudes.

    Differenced from the RAW energy, not the unit-normalized copy: normalization clips outside the
    plotted area, which would flatten the gradient there. It is otherwise affine, and the arrows are
    rescaled by their own max below, so the drawn result is identical either way.
    """
    ys, xs = y[:, 0], x[0]
    dE_dy, dE_dx = np.gradient(energy, ys, xs)      # axis 0 is y, axis 1 is x
    sub = (slice(None, None, stride),) * 2
    px, py = x[sub], y[sub]
    keep = ((px >= lim[0]) & (px <= lim[1]) & (py >= lim[0]) & (py <= lim[1])).ravel()
    pts = np.column_stack([px.ravel(), py.ravel()])[keep]
    d = -np.column_stack([dE_dx[sub].ravel(), dE_dy[sub].ravel()])[keep]
    return pts, d, np.linalg.norm(d, axis=1), stride * float(xs[1] - xs[0])


def quiver_on(ax, arrows, args, color):
    """color: a matplotlib color, or "mag" to color by |grad E| with the cool colormap."""
    pts, d, mag, cell = arrows
    # Longest arrow = one arrow spacing; the rest keep their relative |grad E|.
    d = d * cell / max(mag.max(), 1e-12)
    # Heads scale with the shaft, so without --head-scale a near-zero arrow is all head and reads
    # as a dot. Keep them slim.
    kw = dict(pivot="mid", angles="xy", scale_units="xy", scale=1, width=args.arrow_width,
              headwidth=args.head_scale * 3.0, headlength=args.head_scale * 4.0,
              headaxislength=args.head_scale * 3.5)
    if color == "mag":
        ax.quiver(pts[:, 0], pts[:, 1], d[:, 0], d[:, 1], mag, cmap="cool", **kw)
    else:
        ax.quiver(pts[:, 0], pts[:, 1], d[:, 0], d[:, 1], color=color, **kw)


def frame(fig, rect, args):
    ax = fig.add_axes(rect)
    ax.set_xlim(*args.lim)
    ax.set_ylim(*args.lim)
    ax.set_aspect("equal")
    if not args.axes:
        ax.set_axis_off()
    return ax


def draw(fig, rect, x, y, z, cmap, args, arrows=None):
    ax = frame(fig, rect, args)
    ax.contourf(x, y, z, levels=args.levels, cmap=cmap, vmin=0.0, vmax=1.0)
    if args.line_width > 0:
        ax.contour(x, y, z, levels=args.levels, colors="k", linewidths=args.line_width,
                   alpha=args.line_alpha)
    if arrows is not None:
        quiver_on(ax, arrows, args, args.arrow_color)
    return ax


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", required=True,
                   help='one per model: "Title=dir eval.py --viz-dir wrote to"')
    p.add_argument("--timesteps", type=int, nargs="+", default=[10],
                   help="denoising timesteps to plot (must be among eval's viz timesteps)")
    p.add_argument("--obs", type=int, default=0, help="observation context (_obs{k} in the file name)")
    p.add_argument("--lim", type=float, nargs=2, default=(-6, 6),
                   help="x and y extent, and the area each map is normalized over (match the other panels)")
    p.add_argument("--levels", type=int, default=24, help="number of filled contour levels")
    p.add_argument("--line-width", type=float, default=0.3,
                   help="contour line width (points); 0 = filled bands only")
    p.add_argument("--line-alpha", type=float, default=0.35)
    p.add_argument("--cmap-end", type=float, default=0.8,
                   help="fraction of viridis_r used; < 1 keeps the highest energies off near-black")
    p.add_argument("--arrows", action="store_true", help="also save a copy with the -grad E quiver on top")
    p.add_argument("--arrows-only", action="store_true",
                   help="also save the quiver on white, with no contours (front figure's fig2)")
    p.add_argument("--arrow-stride", type=int, default=7,
                   help="quiver uses every n-th grid point; 7 gives ~17 arrows per axis over a 12-unit lim")
    p.add_argument("--arrow-color", default="0.15",
                   help='any matplotlib color, or "mag" to color by |grad E| with the cool colormap')
    p.add_argument("--arrow-width", type=float, default=0.0035, help="shaft width (fraction of axis width)")
    p.add_argument("--head-scale", type=float, default=0.8,
                   help="arrowhead size relative to matplotlib's default (3/4/3.5 x shaft width)")
    p.add_argument("--axes", action="store_true", help="keep the axes, ticks and frame")
    p.add_argument("--panel-size", type=float, nargs=2, default=(5, 5), help="inches")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = truncated_cmap("viridis_r", 0.0, args.cmap_end)
    w, h = args.panel_size

    for title, viz_dir in (e.split("=", 1) for e in args.dirs):
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        for t in args.timesteps:
            m = np.load(find_map(viz_dir, t, args.obs))
            x, y = m["x"], m["y"]
            z, lo, hi = unit_normalize(x, y, m["energy"], args.lim)
            arrows = gradient_arrows(x, y, m["energy"], args.lim, args.arrow_stride)
            print(f"t={t:3d} {title}: energy over plotted area [{lo:.4g}, {hi:.4g}] -> [0, 1]; "
                  f"{len(arrows[0])} arrows, |grad E| in [{arrows[2].min():.3g}, {arrows[2].max():.3g}]")

            fig = plt.figure(figsize=(w, h))
            draw(fig, [0, 0, 1, 1], x, y, z, cmap, args)
            save(fig, out_dir / f"energy_contour_t{t:03d}_{slug}")

            if args.arrows:
                fig = plt.figure(figsize=(w, h))
                draw(fig, [0, 0, 1, 1], x, y, z, cmap, args, arrows=arrows)
                save(fig, out_dir / f"energy_contour_arrows_t{t:03d}_{slug}")

            if args.arrows_only:
                # On white, colored by |grad E| -- the front figure's fig2.
                fig = plt.figure(figsize=(w, h))
                quiver_on(frame(fig, [0, 0, 1, 1], args), arrows, args, "mag")
                save(fig, out_dir / f"energy_arrows_t{t:03d}_{slug}")


if __name__ == "__main__":
    main()
