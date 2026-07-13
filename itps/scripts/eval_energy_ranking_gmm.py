#!/usr/bin/env python
"""
Check whether a fine-tuned GMM diffusion policy's energy landscape ranks
sampled points the way the ground-truth GMM density does.

For each conditioning context (one for an unconditional GMM policy, one per
cluster for a conditional one):
  1. Sample `--n-samples` candidate points from the ORIGINAL (pre-fine-tune)
     policy.
  2. Score each candidate by its ground-truth density under the preferred
     cluster's Gaussian component (mvn_pdf, from gaussian_mm.py).
  3. Compute the energy of each candidate under the FINE-TUNED policy.
  4. Report the pairwise win rate: over every pair of candidates, how often does
     the (negative) energy order the pair the same way the ground-truth density
     does? Pairs with tied density are thrown out (fraction reported too).

Run from the itps/ directory:
    conda run -n diffpreff python scripts/eval_energy_ranking_gmm.py \\
        --pretrained-path data/gmm/general/train/.../pretrained_model \\
        --finetuned-path  data/gmm/general/tune/.../pretrained_model \\
        --pref-cluster 0 --n-samples 200
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from itps.common.utils.eval_utils import gen_obs
from itps.common.utils.preference_scoring import pairwise_win_rate
from itps.common.utils.utils import set_global_seed
from itps.scripts.gaussian_mm import get_means, get_covs, mvn_pdf


def eval_energy_ranking_gmm(pretrained_policy, finetuned_policy, pref_cluster,
                             conditional=False, n_samples=200, sampler='ddim', seed=None):
    """
    Returns: {"per_context": [{"context_idx", "rho", "pvalue"}, ...],
              "mean_rho": float, "n_contexts_with_variation": int}

    `context_idx` is 0 for an unconditional GMM policy (a single fixed
    observation), or 0/1/2 for a conditional one (one context per cluster,
    matching gen_obs' order).
    """
    if seed is not None:
        set_global_seed(seed)

    means, covs = get_means(), get_covs()
    pref_mean, pref_cov = means[pref_cluster], covs[pref_cluster]
    device = next(pretrained_policy.parameters()).device

    obs_list = gen_obs(conditional=conditional, N=n_samples, device=device)

    per_context = []
    for context_idx, obs in enumerate(obs_list):
        with torch.no_grad():
            actions = pretrained_policy.run_inference(obs, methods=[sampler])
            traj_t = actions[0]  # (n_samples, 1, 2), unnormalized
            energy = finetuned_policy.get_energy(
                action_batch={'action': traj_t}, t=0, observation_batch=obs
            )
        points = traj_t.squeeze(1).cpu().numpy()        # (n_samples, 2)
        density = mvn_pdf(points, pref_mean, pref_cov)   # ground-truth density under the preferred cluster
        neg_energy = -energy.squeeze(-1).cpu().numpy()   # higher = more preferred by the model

        win_rate, n_used, n_tied = pairwise_win_rate(density, neg_energy)
        per_context.append({
            "context_idx": context_idx, "win_rate": win_rate,
            "n_pairs_used": n_used, "n_pairs_tied": n_tied,
        })

    valid = [r["win_rate"] for r in per_context if not np.isnan(r["win_rate"])]
    total_used = sum(r["n_pairs_used"] for r in per_context)
    total_tied = sum(r["n_pairs_tied"] for r in per_context)
    total_pairs = total_used + total_tied
    return {
        "per_context": per_context,
        "mean_win_rate": float(np.mean(valid)) if valid else float('nan'),
        "n_contexts_with_pairs": len(valid),
        "n_pairs_used": total_used,
        "n_pairs_tied": total_tied,
        "pct_pairs_thrown_out": (
            float(100.0 * total_tied / total_pairs) if total_pairs else float('nan')
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pretrained-path", required=True,
                        help="Path to the original (pre-fine-tune) pretrained_model dir")
    parser.add_argument("--finetuned-path", required=True,
                        help="Path to the fine-tuned pretrained_model dir")
    parser.add_argument("--pref-cluster", type=int, required=True, choices=[0, 1, 2],
                        help="Which GMM component is the preferred cluster "
                             "(matches gaussian_mm_pref_data.py's --pref-cluster)")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Candidate points sampled per context (default 200)")
    parser.add_argument("--conditional", action="store_true",
                        help="Use a conditional GMM policy (samples once per cluster "
                             "context instead of once unconditionally)")
    parser.add_argument("--sampler", default="ddim", choices=["ddim", "ired"],
                        help="Sampling method used to generate candidates from the "
                             "pretrained (non-fine-tuned) policy (default: ddim)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default=None, help="Optional .json path to save detailed results")
    args = parser.parse_args()

    pretrained_policy = DiffusionPolicy.from_pretrained(args.pretrained_path)
    finetuned_policy = DiffusionPolicy.from_pretrained(args.finetuned_path)
    for p in (pretrained_policy, finetuned_policy):
        p.to(args.device)
        p.eval()

    results = eval_energy_ranking_gmm(
        pretrained_policy, finetuned_policy, pref_cluster=args.pref_cluster,
        conditional=args.conditional, n_samples=args.n_samples,
        sampler=args.sampler, seed=args.seed,
    )

    n_contexts = 3 if args.conditional else 1
    total_pairs = results['n_pairs_used'] + results['n_pairs_tied']
    print(f"\npref_cluster={args.pref_cluster}  |  {n_contexts} context(s)  |  "
          f"{args.n_samples} candidates/context sampled from the pretrained policy  |  "
          f"energy scored under the fine-tuned policy\n")
    print(f"  mean win rate = {results['mean_win_rate']:.3f}  "
          f"({results['n_contexts_with_pairs']}/{n_contexts} contexts had usable pairs)  |  "
          f"{results['pct_pairs_thrown_out']:.1f}% of pairs thrown out as ties "
          f"({results['n_pairs_tied']}/{total_pairs})")

    if args.save_path is not None:
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed per-context results -> {args.save_path}")


if __name__ == "__main__":
    main()
