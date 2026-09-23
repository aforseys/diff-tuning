#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Generate preference-pair datasets for the GMM environment.

Samples are always drawn from the spec's own (even) blend -- the preference is applied
post-hoc by a utility function `u(x)`, so the training distribution carries no
preference of its own. Points are paired off and the one with higher utility becomes the
winner, which grades preference *within* a cluster as well as between clusters.

Writes `<name>_pos.npy` / `<name>_neg.npy` in the (N, 3) observation format the
dataloader expects, plus a JSON sidecar recording the spec, utility, seed and margin.
"""

import argparse
from pathlib import Path

import numpy as np

from itps.envs.gmm.gaussian_mm import (
    DEFAULT_SPEC,
    GMM_SPECS,
    GMM_UTILITIES,
    get_spec,
    get_utility,
    plot_samples,
    save_observations,
)


PAIRING_MODES = ("any", "same")


def _draw_pairs(spec, utility, n_pairs, seed, margin, pairing):
    """One round: draw 2*n_pairs samples, pair them, and keep the pairs clearing `margin`."""
    sample_seed, shuffle_seed = np.random.SeedSequence(seed).spawn(2)
    X, comps = spec.sample(2 * n_pairs, sample_seed)

    if pairing == "any":
        # Samples are i.i.d., so pairing consecutive draws needs no further randomness.
        left_idx, right_idx = np.arange(0, 2 * n_pairs, 2), np.arange(1, 2 * n_pairs, 2)
    else:
        # Shuffle within each cluster and split it in half, so both members of a pair
        # come from the same cluster. A separate stream from the one spec.sample used.
        rng = np.random.default_rng(shuffle_seed)
        left_parts, right_parts = [], []
        for k in range(spec.n_clusters):
            idx = np.where(comps == k)[0]
            rng.shuffle(idx)
            half = len(idx) // 2
            left_parts.append(idx[:half])
            right_parts.append(idx[half:2 * half])
        left_idx, right_idx = np.concatenate(left_parts), np.concatenate(right_parts)
        # Shuffle the pair order too, so which cluster a pair comes from is random rather than
        # ordered by cluster -- otherwise keeping the first n_pairs would favour cluster 0.
        order = rng.permutation(len(left_idx))
        left_idx, right_idx = left_idx[order], right_idx[order]

    left, right = X[left_idx], X[right_idx]
    left_comps, right_comps = comps[left_idx], comps[right_idx]
    u_left, u_right = utility(left), utility(right)

    keep = np.abs(u_left - u_right) > margin
    left, right = left[keep], right[keep]
    left_comps, right_comps = left_comps[keep], right_comps[keep]
    u_left, u_right = u_left[keep], u_right[keep]

    left_wins = (u_left > u_right)[:, None]
    return len(keep), {
        'winners': np.where(left_wins, left, right),
        'losers': np.where(left_wins, right, left),
        'winner_comps': np.where(left_wins[:, 0], left_comps, right_comps),
        'loser_comps': np.where(left_wins[:, 0], right_comps, left_comps),
        'gaps': np.abs(u_left - u_right),
    }


def generate_preference_pairs(spec, utility, n_pairs, seed, margin=0.0, pairing="any", max_rounds=100):
    """
    Draw samples from `spec`, pair them off, and label each pair by `utility`. Returns exactly
    `n_pairs` pairs, drawing repeatedly until that many clear the margin.

    pairing: how the two members of a pair are chosen.
        "any"  -- drawn independently, so a pair may straddle two clusters, and most of the
                  preference signal is then cross-mode.
        "same" -- both members come from the same cluster, so all of the signal is the utility
                  spread *within* a mode. Required for a conditional policy, where the two
                  members must share a conditioning context for the comparison to be well posed.
    margin: pairs whose utilities differ by <= margin are discarded and redrawn, since the
        preference between them is not meaningful. Raising it gives cleaner pairs at the cost of
        more draws; the returned dict reports how many were drawn in total.

    Returns both observation formats of the winners and losers, in the same (N, 3) layout
    the dataloader reads: 'unconditional_*' zeroes the observation column, 'conditional_*'
    carries each sample's own cluster index. With pairing="same" a pair's two members
    share that index; with "any" they generally do not, which is why "any" data belongs
    with an unconditional policy.
    """
    if pairing not in PAIRING_MODES:
        raise ValueError(f"Unknown pairing {pairing!r}. Available: {list(PAIRING_MODES)}")

    # Draw repeatedly until n_pairs survive the margin filter, then keep exactly that many.
    parts = {k: [] for k in ("winners", "losers", "winner_comps", "loser_comps", "gaps")}
    n_drawn = 0
    for round_i in range(max_rounds):
        drawn, kept = _draw_pairs(spec, utility, n_pairs, [seed, round_i], margin, pairing)
        n_drawn += drawn
        for k, v in kept.items():
            parts[k].append(v)
        if sum(len(v) for v in parts["winners"]) >= n_pairs:
            break
    else:
        raise RuntimeError(
            f"Only {sum(len(v) for v in parts['winners'])} of {n_pairs} pairs survived margin={margin} "
            f"after {max_rounds} rounds ({n_drawn} pairs drawn). Lower the margin."
        )

    winners, losers, winner_comps, loser_comps, gaps = (
        np.concatenate(parts[k])[:n_pairs] for k in ("winners", "losers", "winner_comps", "loser_comps", "gaps")
    )

    def observations(points, cluster_idx, conditional):
        column = cluster_idx[:, None].astype(float) if conditional else np.zeros((len(points), 1))
        return np.hstack([column, points])

    return {
        'unconditional_positive': observations(winners, winner_comps, False),
        'unconditional_negative': observations(losers, loser_comps, False),
        'conditional_positive': observations(winners, winner_comps, True),
        'conditional_negative': observations(losers, loser_comps, True),
        'pairing': pairing,
        'n_pairs': int(len(winners)),
        'n_pairs_drawn': int(n_drawn),
        'mean_utility_gap': float(gaps.mean()),
        'winner_cluster_counts': np.bincount(winner_comps, minlength=spec.n_clusters).tolist(),
        'same_cluster_pair_rate': float((winner_comps == loser_comps).mean()),
    }


def plot_pref_dataset(pref_dataset):
    # Don't plot observation (first element of 2nd dimension)
    plot_samples(pref_dataset['unconditional_positive'][:, 1:])
    plot_samples(pref_dataset['unconditional_negative'][:, 1:])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gmm-spec", type=str, default=DEFAULT_SPEC, choices=sorted(GMM_SPECS),
                        help=f"Which registered GMM to sample from (default: {DEFAULT_SPEC})")
    parser.add_argument("--utility", type=str, default="diagonal", choices=sorted(GMM_UTILITIES),
                        help="Preference utility used to label pairs (default: diagonal)")
    parser.add_argument("--n-pairs", type=int, default=1000, help="Number of pairs to draw (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--pairing", type=str, default="any", choices=PAIRING_MODES,
                        help="'any': a pair's two samples may come from different clusters "
                             "(for an unconditional policy). 'same': both come from the same "
                             "cluster, so the pair shares a conditioning context.")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Drop pairs whose utilities differ by <= this (default: 0.0)")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Directory to save the pair datasets (default: no saving)")
    parser.add_argument("--name", type=str, default=None,
                        help="Basename for the saved files (default: gmm_<spec>_<utility>_<n>_<seed>)")
    parser.add_argument("--viz", action="store_true", help="Show winner/loser scatter plots")
    args = parser.parse_args()

    spec = get_spec(args.gmm_spec)
    utility = get_utility(args.utility)
    pref_dataset = generate_preference_pairs(
        spec, utility, args.n_pairs, args.seed, margin=args.margin, pairing=args.pairing
    )

    print(f"pairing={args.pairing}: {pref_dataset['n_pairs']} pairs from "
          f"{pref_dataset['n_pairs_drawn']} drawn at margin {args.margin}; "
          f"mean utility gap {pref_dataset['mean_utility_gap']:.4f}")
    print(f"winners per cluster: {pref_dataset['winner_cluster_counts']}  |  "
          f"{100 * pref_dataset['same_cluster_pair_rate']:.0f}% of pairs are within one cluster")

    if args.viz:
        plot_pref_dataset(pref_dataset)

    if args.save_path is not None:
        save_dir = Path(args.save_path)
        base = args.name or f"gmm_{spec.name}_{args.utility}_{args.pairing}_{args.n_pairs}_{args.seed}"
        meta = dict(
            spec=spec.as_dict(), utility=args.utility, seed=args.seed, margin=args.margin,
            pairing=args.pairing,
            n_pairs=pref_dataset['n_pairs'], n_pairs_drawn=pref_dataset['n_pairs_drawn'],
            mean_utility_gap=pref_dataset['mean_utility_gap'],
            winner_cluster_counts=pref_dataset['winner_cluster_counts'],
            same_cluster_pair_rate=pref_dataset['same_cluster_pair_rate'],
        )
        # Both observation formats, from the same pairs -- pick the one matching the
        # policy's condition_type.
        for kind in ("unconditional", "conditional"):
            for role, key in (("pos", f'{kind}_positive'), ("neg", f'{kind}_negative')):
                path = save_observations(pref_dataset[key], save_dir / f"{base}_{kind}_{role}.npy",
                                         role=role, condition_type=kind, **meta)
                print(f"Saved {kind} {role} dataset to {path}")
