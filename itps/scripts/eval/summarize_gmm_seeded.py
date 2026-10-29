#!/usr/bin/env python
"""
Summarize the final seeded GMM runs, one CSV per method, as mean ± standard error over seeds.

For each CSV passed, reports:
  mean utility        of the method's own samples (absolute, not relative to the base model)
  normalized utility  the same, rescaled to the utility range of the training data, one shared range
                      for every method: (U - lo) / (hi - lo), lo/hi = the min/max utility of the
                      --training-samples points. 0 = the worst training point, 1 = the best; above 1
                      means the method's average sample beats the best training point.
  delta LL            log-likelihood of the method's own samples minus the base model's DDIM samples,
                      paired by seed

The method's sampler is IRED for an EBM run (policy.name == ebm_diffusion, using its single
eval.opt_params setting) and DDIM otherwise. The base evals only sample with DDIM, so for the EBM
method delta LL = LL(finetuned, IRED) - LL(base, DDIM); for the DPO methods it is
LL(finetuned, DDIM) - LL(base, DDIM).

Inputs
  --runs-csv  One wandb "Export CSV" of the runs table per method (e.g. project gmm-final-seeded,
              filtered to one method). Each row is one run, and its eval/* columns hold that run's LAST
              logged eval, which is checked to be the final step (eval/step == training.offline_steps).
  --base-dir  Base-model evals made with scripts/eval/eval.py, one folder per seed:
                <base-dir>/dp_seed{k}/eval_info.json   plain-diffusion base (for the DPO methods)
                <base-dir>/ebm_seed{k}/eval_info.json  EBM base (for the energy method)

Processing
  1. Runs are keyed by their `seed` column (a repeated seed is an error).
  2. delta_k = LL(finetuned seed k) - LL(base seed k); both evals of seed k draw their samples from
     the same seed, so pairing cancels shared sampling noise.
  3. Mean ± SE over seeds, SE = std(ddof=1) / sqrt(n_seeds).

Usage (from itps/):
  python scripts/eval/summarize_gmm_seeded.py --base-dir outputs/base_ref \
      --training-samples data/gmm_obs/gmm_cluster_line_unconditional_1000_0_20260918_122402.npy \
      --runs-csv ebm.csv dpo_forward.csv dpo_original.csv [--per-seed]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from itps.envs.gmm.gaussian_mm import get_utility


def to_float(value):
    return float(value) if value not in ("", None) else np.nan


def mean_se(values):
    v = np.asarray(values, dtype=float)
    se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return f"{v.mean():+.3f} ± {se:.3f}"


def load_base(base_dir, prefix, seed):
    path = Path(base_dir) / f"{prefix}_seed{seed}" / "eval_info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing base eval for seed {seed}: {path}")
    return json.loads(path.read_text())["aggregated"]


def summarize(csv_path, base_dir, utility_range, per_seed):
    rows = list(csv.DictReader(open(csv_path, newline="")))
    by_seed = {}
    for row in rows:
        seed = int(float(row["seed"]))
        if seed in by_seed:
            raise ValueError(f"{csv_path}: two runs with seed {seed}; export one run per seed.")
        if to_float(row["eval/step"]) != to_float(row["training.offline_steps"]):
            print(f"WARNING: {row['Name']} last eval is step {row['eval/step']}, not the final "
                  f"{row['training.offline_steps']}; its values are not the final model's.")
        by_seed[seed] = row
    seeds = sorted(by_seed)

    is_ebm = by_seed[seeds[0]]["policy.name"] == "ebm_diffusion"
    if is_ebm:
        ired = sorted(c for c in rows[0] if c.startswith("eval/IRED_") and c.endswith("_log_likelihood"))
        if len(ired) != 1:
            raise ValueError(f"{csv_path}: expected exactly one IRED setting, found {ired}")
        label = ired[0][len("eval/"):-len("_log_likelihood")]   # e.g. IRED_1steps_last5_denoise
        ll_key, util_key, base_prefix = ired[0], f"eval/mean_utility_{label}", "ebm"
    else:
        label, ll_key, util_key, base_prefix = "DDIM", "eval/DDIM_log_likelihood", "eval/mean_utility_DDIM", "dp"

    base = {k: load_base(base_dir, base_prefix, k) for k in seeds}
    utility = np.array([to_float(by_seed[k][util_key]) for k in seeds])
    ll = np.array([to_float(by_seed[k][ll_key]) for k in seeds])
    base_ll = np.array([base[k]["DDIM_log_likelihood"] for k in seeds])
    delta_ll = ll - base_ll
    lo, hi = utility_range
    norm_utility = (utility - lo) / (hi - lo)

    print(f"\n=== {Path(csv_path).name}: {len(seeds)} seeds {seeds}, sampler {label}, base {base_prefix}_seed{{k}} (DDIM) ===")
    print(f"  mean utility        {mean_se(utility)}")
    print(f"  normalized utility  {mean_se(norm_utility)}   (training-data range [{lo:+.3f}, {hi:+.3f}])")
    print(f"  delta LL            {mean_se(delta_ll)}")
    if per_seed:
        for k, u, n, l, b in zip(seeds, utility, norm_utility, ll, base_ll):
            print(f"    seed {k}: utility {u:+.3f} (normalized {n:.3f}) | LL {l:+.3f} - base {b:+.3f} = {l - b:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-csv", nargs="+", required=True, help="one wandb runs-table CSV export per method")
    parser.add_argument("--base-dir", required=True, help="folder with {dp,ebm}_seed{k}/eval_info.json")
    parser.add_argument("--training-samples", required=True,
                        help="the (N, 3) .npy the base models were trained on; its utility range normalizes utility")
    parser.add_argument("--utility", default="diagonal", help="registered GMM utility (default: diagonal)")
    parser.add_argument("--per-seed", action="store_true", help="also print every seed's values")
    args = parser.parse_args()
    data_utility = get_utility(args.utility)(np.load(args.training_samples)[:, 1:])
    utility_range = (float(data_utility.min()), float(data_utility.max()))
    for csv_path in args.runs_csv:
        summarize(csv_path, args.base_dir, utility_range, args.per_seed)


if __name__ == "__main__":
    main()
