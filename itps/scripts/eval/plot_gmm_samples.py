#!/usr/bin/env python
"""
Scatter plots of GMM samples colored by preference utility, one figure per sample set, for building
a panel: the training samples, plus the generated samples of each method.

Every figure shares the same axes, the same color scale (the training data's utility range, so
colors match the normalized utility) and dashed outlines of the true GMM clusters at --n-std
standard deviations. Samples beyond the color range get the end color. Each figure has the colorbar
on its right, labeled --cbar-label.

Generated samples come from the .npz that `scripts/eval/eval.py --save-samples <path>` writes, whose
keys are sampler names ("ddim", "ired_<settings>"), each an (N, 2) array.

Usage (from itps/):
  python scripts/eval/plot_gmm_samples.py \
      --training-samples data/gmm_obs/gmm_cluster_line_unconditional_1000_0_20260918_122402.npy \
      --samples "DPO (Forward KL)=outputs/samples/dpo_forward_seed0.npz:ddim" \
                "DPO (Reverse KL)=outputs/samples/dpo_reverse_seed0.npz:ddim" \
                "Ours=outputs/samples/ours_seed0.npz:ired_1steps_last5_denoise" \
      --out-dir outputs/figures/samples
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from itps.envs.gmm.gaussian_mm import draw_cluster_outlines, get_spec, get_utility

plt.rcParams.update({
    # Palatino, with close substitutes for systems that don't ship it.
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "TeX Gyre Pagella", "URW Palladio L", "Book Antiqua"],
    # Math text (e.g. a "$r^*$" colorbar label) in Palatino too, not matplotlib's default math font.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Palatino",
    "mathtext.it": "Palatino:italic",
    "mathtext.bf": "Palatino:bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})
TITLE_SIZE = 24
AXIS_LABEL_SIZE = 26  # x, y and the colorbar label
# matplotlib only finds Palatino's regular face, so with LaTeX available, render all text through it
# with mathpazo (LaTeX's Palatino) to get a true Palatino italic for math like "$r^*$".
if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{mathpazo}"})


def finish_panel(fig, ax, title, spec, norm, cmap, n_std, lim, cbar_label, path):
    """Shared styling for every GMM panel (samples and D-MSE maps): colorbar on the right, true
    cluster outlines, equal axes with matching ticks, labels and title. Saves and closes the figure."""
    # Colorbar axis sized to the plot's height (equal aspect would otherwise mismatch them).
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    # Built from the norm rather than the plotted artist, so the bar isn't faded by any alpha.
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(cbar_label, fontsize=AXIS_LABEL_SIZE)
    cax.tick_params(labelsize=14)
    draw_cluster_outlines(ax, spec, n_std)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    # Same ticks on both axes (with large fonts matplotlib would otherwise pick different steps).
    ticks = np.linspace(lim[0], lim[1], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("y", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_samples(points, title, spec, utility, norm, cmap, n_std, lim, cbar_label, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(points[:, 0], points[:, 1], c=utility(points), cmap=cmap, norm=norm,
               s=10, alpha=0.8, edgecolor="none", zorder=2)
    finish_panel(fig, ax, title, spec, norm, cmap, n_std, lim, cbar_label, path)
    outside = np.sum(np.any((points < lim[0]) | (points > lim[1]), axis=1))
    note = f" ({outside} of {len(points)} samples fall outside the axes)" if outside else ""
    print(f"Saved {path}{note}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-samples", required=True, help="(N, 3) .npy the base models were trained on")
    parser.add_argument("--samples", nargs="*", default=[],
                        help='one per method: "Title=path/to/samples.npz:sampler_key"')
    parser.add_argument("--gmm-spec", default="cluster_line")
    parser.add_argument("--utility", default="diagonal")
    parser.add_argument("--n-std", type=float, default=2.0, help="cluster outline radius in standard deviations")
    parser.add_argument("--lim", type=float, nargs=2, default=(-10, 10), help="x and y axis limits")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--cbar-label", default="reward ($r^*$)", help='colorbar label (default: "reward ($r^*$)")')
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    spec, utility = get_spec(args.gmm_spec), get_utility(args.utility)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = np.load(args.training_samples)[:, 1:]
    train_u = utility(train)
    # One color scale for every figure: the training data's utility range.
    norm = Normalize(vmin=train_u.min(), vmax=train_u.max(), clip=True)
    print(f"Color scale (training-data utility range): [{norm.vmin:.4f}, {norm.vmax:.4f}]")

    plot_samples(train, "Training samples", spec, utility, norm, args.cmap, args.n_std, args.lim,
                 args.cbar_label, out_dir / "training_samples.png")
    for entry in args.samples:
        title, rest = entry.split("=", 1)
        path, key = rest.rsplit(":", 1)
        points = np.load(path)[key]
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        plot_samples(points, title, spec, utility, norm, args.cmap, args.n_std, args.lim,
                     args.cbar_label, out_dir / f"{slug}.png")


if __name__ == "__main__":
    main()
