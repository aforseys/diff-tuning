#!/usr/bin/env python
"""
3D energy surfaces of several EBMs stacked top to bottom, one figure per denoising timestep, in the
style of the front figure (notebooks/front_figure.ipynb): no axes or labels, viridis_r stopped at 0.8
of its range, elevation 2, azimuth -60. For depth, the surface is lit softly from one side and carries
a thin mesh.

Reads the .npz that `scripts/eval/eval.py --viz --viz-dir <dir>` writes for every viz timestep
(<dir>/energy_surface_s{k}_t{ttt}_obs0.npz, keys x, y, energy) and normalizes each map exactly as
the 2D panels do (plot_gmm_dmse.py --key energy --norm unit): min-max to [0,1] over the plotted
area, per map. Every panel then shares the same [0,1] height and color scale. Lower energy = the
model considers that point more likely, so likely regions are the bright valleys.

A dot marks each GMM cluster mean, at the surface's height there, colored by its reward on the same
scale as the sampling figure (plot_gmm_samples.py: the training data's utility range) but with its
own colormap (RdPu by default). Rewards depend only on position, so each dot keeps its color in every
panel; only its height changes. Dots are always drawn on top of the surface.

The --cbar-timestep figure (default t=0, fully denoised) also gets the energy colorbar
and the reward colorbar, both on its --cbar-side (default right), styled like the sampling figure's. Its panels are the same
size as every other figure's; the bars widen the figure instead.

matplotlib cannot clip 3D surfaces to the axes, so the grid is cropped to --lim before drawing
(the 2D panels clip instead); the normalized values inside the area are identical.

Usage (from itps/):
  python scripts/eval/plot_gmm_energy_3d.py \
      --dirs "Base (EBM)=outputs/energy/ebm_base" "Ours=outputs/energy/ebm_seed0" \
      --training-samples data/gmm_obs/gmm_cluster_line_unconditional_1000_0_20260918_122402.npy \
      --out-dir outputs/figures/energy_3d
"""

import argparse
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, Normalize
from scipy.interpolate import RegularGridInterpolator

# Same per-map normalization as the 2D energy panels, and the sampling figure's fonts and label size
# (imported from the scripts next to this one; importing plot_gmm_samples also sets its rcParams).
from plot_gmm_dmse import unit_normalize
from plot_gmm_samples import AXIS_LABEL_SIZE

from itps.envs.gmm.gaussian_mm import get_spec, get_utility


def find_map(viz_dir, t, obs):
    """The energy map eval.py saved for timestep t. Matched on t, not on the s{k} index (s9 is t000)."""
    matches = sorted(Path(viz_dir).glob(f"energy_surface_s*_t{t:03d}_obs{obs}.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one energy_surface_s*_t{t:03d}_obs{obs}.npz in {viz_dir}, "
                                f"found {len(matches)}")
    return matches[0]


def crop(x, y, z, lim):
    """The part of the (meshgrid) map inside lim x lim."""
    cols = (x[0] >= lim[0]) & (x[0] <= lim[1])
    rows = (y[:, 0] >= lim[0]) & (y[:, 0] <= lim[1])
    keep = np.ix_(rows, cols)
    return x[keep], y[keep], z[keep]


def surface_height(x, y, z, points):
    """z (on the meshgrid x, y) at each (x, y) point, bilinearly interpolated so a dot sits on the surface."""
    return RegularGridInterpolator((y[:, 0], x[0]), z)(points[:, ::-1])


def truncated_cmap(name, start, end):
    return ListedColormap(plt.get_cmap(name)(np.linspace(start, end, 256)))


def draw_surface(fig, rect, x, y, z, cmap, dots, args):
    """dots: (points (K, 2), heights (K,), colors (K, 4))."""
    ax = fig.add_axes(rect, projection="3d")
    # Plain draw order instead of matplotlib's depth sort, which places each whole collection (all
    # the dots at once) either in front of or behind the whole surface: dots always on top.
    ax.computed_zorder = False
    # Colors: the energy colormap lit from one side, so slopes read as 3D. --shade < 1 softens the
    # lighting, keeping the colors close to the energy colorbar. vert_exag makes the lighting see the
    # surface as drawn: the default 4:4:3 box puts the lim span on 4 units and z's [0,1] on 3.
    light = LightSource(azdeg=args.light_az, altdeg=35)
    colors = light.shade(z, cmap=cmap, vmin=0.0, vmax=1.0, blend_mode="soft", fraction=args.shade,
                         vert_exag=0.75 * (args.lim[1] - args.lim[0]), dx=x[0, 1] - x[0, 0], dy=y[1, 0] - y[0, 0])
    # Mesh: a coarse grid of thin dark lines along the surface. They are the patches' edges, so the
    # surface hides them wherever it hides itself.
    ax.plot_surface(x, y, z, facecolors=colors, rcount=args.mesh_count, ccount=args.mesh_count,
                    edgecolor=(0.0, 0.0, 0.0, args.mesh_alpha), linewidth=args.mesh_width, shade=False,
                    zorder=1)
    points, heights, colors = dots
    ax.scatter(points[:, 0], points[:, 1], heights, c=colors, s=args.dot_size, edgecolors="black",
               linewidths=args.dot_edge, depthshade=False, zorder=2)
    ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect(None, zoom=args.zoom)
    ax.set_axis_off()


def add_colorbar(fig, rect, norm, cmap, label, side):
    """A colorbar styled like the sampling figure's, with its ticks and label on `side`."""
    cax = fig.add_axes(rect)
    # Built from the norm rather than a plotted artist, as in plot_gmm_samples.finish_panel.
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(label, fontsize=AXIS_LABEL_SIZE)
    cax.tick_params(labelsize=14)
    cax.yaxis.set_ticks_position(side)
    cax.yaxis.set_label_position(side)


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {path}.png/.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", required=True,
                        help='one per model, top to bottom: "Title=dir eval.py --viz-dir wrote to"')
    parser.add_argument("--training-samples", required=True,
                        help="(N, 3) .npy the base models were trained on; its utility range sets the "
                             "reward color scale, as in plot_gmm_samples.py")
    parser.add_argument("--gmm-spec", default="cluster_line", help="the dots go at this spec's cluster means")
    parser.add_argument("--utility", default="diagonal")
    parser.add_argument("--timesteps", type=int, nargs="+", default=list(range(0, 100, 10)),
                        help="denoising timesteps to plot (must be among eval's viz timesteps)")
    parser.add_argument("--obs", type=int, default=0, help="observation context (_obs{k} in the file name)")
    parser.add_argument("--lim", type=float, nargs=2, default=(-6, 6),
                        help="x and y extent, and the area each map is normalized over (match the 2D panels)")
    parser.add_argument("--elev", type=float, default=2.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--zoom", type=float, default=1.21,
                        help="1.21 reproduces the front figure's camera (matplotlib 3.5 dist=8.2)")
    parser.add_argument("--cmap-end", type=float, default=0.8,
                        help="fraction of viridis_r used; < 1 keeps the highest energies off near-black")
    parser.add_argument("--shade", type=float, default=0.5,
                        help="lighting strength; 0 = flat colormap colors, 1 = full hillshade")
    parser.add_argument("--light-az", type=float, default=270.0, help="light direction (LightSource azdeg)")
    parser.add_argument("--mesh-count", type=int, default=40, help="mesh lines per direction")
    parser.add_argument("--mesh-alpha", type=float, default=0.45, help="mesh line opacity")
    parser.add_argument("--mesh-width", type=float, default=0.3, help="mesh line width (points); 0 = no mesh")
    parser.add_argument("--reward-cmap", default="RdPu")
    parser.add_argument("--reward-cmap-range", type=float, nargs=2, default=(0.3, 1.0),
                        help="part of --reward-cmap used; starting above 0 keeps low rewards off near-white")
    parser.add_argument("--dot-size", type=float, default=330.0, help="scatter marker area (points^2)")
    parser.add_argument("--dot-edge", type=float, default=0.8, help="dot outline width (points)")
    parser.add_argument("--cbar-timestep", type=int, default=0, help="the one timestep whose figure gets colorbars")
    parser.add_argument("--energy-label", default="normalized energy")
    parser.add_argument("--reward-label", default="reward ($r^*$)")
    parser.add_argument("--panel-size", type=float, nargs=2, default=(6, 5), help="inches per panel")
    parser.add_argument("--cbar-side", default="right", choices=["left", "right"],
                        help="side of the panels both colorbars go on")
    parser.add_argument("--cbar-margin", type=float, default=1.4,
                        help="inches added to the colorbar figure per bar, for the bar, its ticks and label")
    parser.add_argument("--single-panels", action="store_true", help="also save each model's panel on its own")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = truncated_cmap("viridis_r", 0.0, args.cmap_end)
    models = [entry.split("=", 1) for entry in args.dirs]
    w, h = args.panel_size

    # Dots: the cluster means, colored by reward on the sampling figure's scale.
    means = np.array(get_spec(args.gmm_spec).means, dtype=float)
    utility = get_utility(args.utility)
    train_u = utility(np.load(args.training_samples)[:, 1:])
    reward_norm = Normalize(vmin=train_u.min(), vmax=train_u.max(), clip=True)
    reward_cmap = truncated_cmap(args.reward_cmap, *args.reward_cmap_range)
    dot_colors = reward_cmap(reward_norm(utility(means)))
    print(f"Reward color scale (training-data utility range): [{reward_norm.vmin:.4f}, {reward_norm.vmax:.4f}]")

    for t in args.timesteps:
        panels = []
        for title, viz_dir in models:
            m = np.load(find_map(viz_dir, t, args.obs))
            z_unit, lo, hi = unit_normalize(m["x"], m["y"], m["energy"], args.lim)
            print(f"t={t:3d} {title}: energy range over plotted area [{lo:.4g}, {hi:.4g}] -> [0, 1]")
            dots = (means, surface_height(m["x"], m["y"], z_unit, means), dot_colors)
            panels.append((title, *crop(m["x"], m["y"], z_unit, args.lim), dots))

        n = len(panels)
        # (norm, cmap, label) of each colorbar, left to right; only the --cbar-timestep figure has them.
        bars = [(Normalize(0.0, 1.0), cmap, args.energy_label), (reward_norm, reward_cmap, args.reward_label)]
        if t != args.cbar_timestep:
            bars = []
        slot = args.cbar_margin
        panel_x = slot * len(bars) if args.cbar_side == "left" else 0.0
        fig_w = w + slot * len(bars)
        fig = plt.figure(figsize=(fig_w, h * n))
        for i, (_, x, y, z, dots) in enumerate(panels):
            draw_surface(fig, [panel_x / fig_w, 1 - (i + 1) / n, w / fig_w, 1 / n], x, y, z, cmap, dots, args)
        bar_w, gap, bar_h = 0.2, 0.15, 0.6  # inches, inches, fraction of the figure height
        for k, (norm, bar_cmap, label) in enumerate(bars):
            # Each bar sits at the panel-facing edge of its slot, its ticks and label facing away.
            x0 = w + k * slot + gap if args.cbar_side == "right" else (k + 1) * slot - gap - bar_w
            add_colorbar(fig, [x0 / fig_w, (1 - bar_h) / 2, bar_w / fig_w, bar_h], norm, bar_cmap, label,
                         args.cbar_side)
        save(fig, out_dir / f"energy3d_t{t:03d}")

        if args.single_panels:
            for title, x, y, z, dots in panels:
                fig = plt.figure(figsize=(w, h))
                draw_surface(fig, [0, 0, 1, 1], x, y, z, cmap, dots, args)
                slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
                save(fig, out_dir / f"energy3d_t{t:03d}_{slug}")


if __name__ == "__main__":
    main()
