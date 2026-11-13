#!/usr/bin/env python
"""
Denoising-MSE (D-MSE) maps of several models on ONE shared log color scale, styled like
plot_gmm_samples.py (same fonts, axes, ticks and true-cluster outlines) for building a panel.

Each map comes from the .npz that `scripts/eval/eval.py --viz-dmse --viz-dir <dir>` writes
(<dir>/denoising_mse_obs0.npz, keys x, y, dmse). All maps use the same seeded (t, eps) draws, so
they are directly comparable. The shared color range is the lowest and highest D-MSE over all maps
within the plotted area, so a color means the same error in every panel. Lower D-MSE = the model
considers that point more likely.

Also plots the energy maps that `--viz --viz-dir <dir>` writes (<dir>/energy_surface_s{k}_t{ttt}_obs0.npz,
keys x, y, energy), via --key energy --norm unit. Energy needs the per-map normalization because
E = -sqrt(1 - alpha_bar_t) * log p_t + c(t): both the scale and the offset are model- and
timestep-specific, and min-max is invariant to that affine pair, so unit-normalized panels are
comparable across models AND timesteps. A shared absolute scale is not.

Usage (from itps/):
  python scripts/eval/plot_gmm_dmse.py --lim -6 6 \
      --maps "Base (diffusion)=outputs/dmse/dp_base/denoising_mse_obs0.npz" \
             "DPO (Forward KL)=outputs/dmse/dpo_forward_seed0/denoising_mse_obs0.npz" \
             "Ours=outputs/dmse/ebm_seed0/denoising_mse_obs0.npz" \
      --out-dir outputs/figures/dmse

  python scripts/eval/plot_gmm_dmse.py --lim -6 6 --key energy --norm unit \
      --cbar-label "normalized energy" \
      --maps "Base (EBM)=outputs/energy/ebm_base/energy_surface_s9_t000_obs0.npz" \
             "Ours=outputs/energy/ebm_seed0/energy_surface_s9_t000_obs0.npz" \
      --out-dir outputs/figures/energy
"""

import argparse
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# Shared fonts, sizes and panel styling (imported from the script next to this one).
from plot_gmm_samples import finish_panel

from itps.envs.gmm.gaussian_mm import get_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", nargs="+", required=True, help='one per model: "Title=path/to/denoising_mse_obs0.npz"')
    parser.add_argument("--key", default="dmse", help="array to plot inside each .npz (dmse or energy)")
    parser.add_argument("--norm", default="shared-log", choices=["shared-log", "unit"],
                        help="shared-log: one absolute log scale over all maps (D-MSE). "
                             "unit: min-max each map to [0,1] over the plotted area (energy)")
    parser.add_argument("--gmm-spec", default="cluster_line")
    parser.add_argument("--n-std", type=float, default=2.0, help="cluster outline radius in standard deviations")
    parser.add_argument("--lim", type=float, nargs=2, default=(-6, 6), help="x and y axis limits")
    parser.add_argument("--cmap", default="viridis_r", help="reversed so low D-MSE (likely) is bright")
    parser.add_argument("--cbar-label", default="D-MSE")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    spec = get_spec(args.gmm_spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = []
    for entry in args.maps:
        title, path = entry.split("=", 1)
        d = np.load(path)
        maps.append((title, d["x"], d["y"], d[args.key]))

    def visible(x, y, z):
        inside = (x >= args.lim[0]) & (x <= args.lim[1]) & (y >= args.lim[0]) & (y <= args.lim[1])
        return z[inside]

    if args.norm == "unit":
        # Each map min-max normalized over the plotted area, NOT over the full grid: the corners
        # are extrapolation far from any data, and letting them set the denominator makes it swing
        # between models for reasons unrelated to the landscape. Values outside the area are clipped
        # so the [0,1] colorbar stays honest.
        rescaled = []
        for title, x, y, z in maps:
            v = visible(x, y, z)
            lo, hi = float(v.min()), float(v.max())
            rescaled.append((title, x, y, np.clip((z - lo) / max(hi - lo, 1e-12), 0.0, 1.0)))
            print(f"{title}: {args.key} range over plotted area [{lo:.4g}, {hi:.4g}] -> [0, 1]")
        maps = rescaled
        norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        # One color scale for every map: min/max over all maps, within the plotted area.
        vmin = min(visible(x, y, z).min() for _, x, y, z in maps)
        vmax = max(visible(x, y, z).max() for _, x, y, z in maps)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        print(f"Shared {args.cbar_label} color scale (log): [{vmin:.4g}, {vmax:.4g}]")

    for title, x, y, z in maps:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(z, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                  cmap=args.cmap, norm=norm, interpolation="nearest", zorder=1)
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        path = out_dir / f"{args.key}_{slug}.png"
        finish_panel(fig, ax, title, spec, norm, args.cmap, args.n_std, args.lim, args.cbar_label, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
