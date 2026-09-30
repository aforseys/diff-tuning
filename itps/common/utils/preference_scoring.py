"""
Environment-agnostic preference-ranking math.

Nothing here knows about mazes, GMM clusters, or robosuite bins — it just
compares two per-sample arrays (a preference metric and some other value,
e.g. model energy) and reports how well their rankings agree. Environment-
specific scoring functions (e.g. `maze_scoring.py`) produce the `scores`
array; this module only does the correlation.
"""

import numpy as np
from scipy.stats import spearmanr


def rank_correlation(scores, values):
    """
    Spearman's rank correlation between `scores` and `values`.

    Use this to check whether a model's energy landscape (pass `values =
    -energy`, since lower energy = more preferred) agrees with a preference
    metric's ranking. Spearman ranks both arrays (tied values, e.g. a binary
    metric like collision_rate, get averaged ranks) and correlates the ranks,
    so no threshold/tie-handling is needed by the caller.

    scores: (N,) array-like — the preference metric, higher = more preferred.
    values: (N,) array-like — the value being checked against `scores`'s
        ranking, higher = more preferred.

    Returns: (rho, pvalue), both `nan` if either array is constant (no
        variation to rank, e.g. every sample collided or none did).
    """
    scores = np.asarray(scores, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.all(scores == scores[0]) or np.all(values == values[0]):
        return float('nan'), float('nan')
    rho, pvalue = spearmanr(scores, values)
    return float(rho), float(pvalue)


def pairwise_win_rate(scores, values, tie_tol=0.0, n_pairs=None, seed=0):
    """
    Classic pairwise win rate: over every unordered pair of samples, how often
    does `values` order the pair the same way `scores` (the ground-truth
    preference) does?

    Pairs whose ground-truth scores are tied (|scores_i - scores_j| <= tie_tol)
    are thrown out, since there is no ground-truth winner to agree with. Use
    this instead of `rank_correlation` when you want a directly interpretable
    "the model agrees with the ground truth on X% of decidable comparisons".

    scores: (N,) ground-truth preference, higher = more preferred.
    values: (N,) quantity being checked (higher = more preferred by the model;
        pass values = -energy so lower energy counts as "model prefers").
    tie_tol: pairs with |score difference| <= this are discarded as ties
        (default 0.0 = only exactly-equal ground-truth scores are thrown out).
        Set it to the margin the training pairs were generated with to score the
        model on comparisons of the same difficulty it was trained on.
    n_pairs: score a random subset of this many pairs instead of all N*(N-1)/2.
        Drawn from the non-tied pairs, so n_pairs_used is exactly n_pairs unless
        fewer than that survive the tie filter. None uses every pair.
    seed: fixes which pairs the subsample draws; ignored when n_pairs is None.

    Returns: (win_rate, n_pairs_used, n_pairs_tied), where
        win_rate: fraction of the kept (non-tied) pairs the model orders
            correctly. A pair where the model's values are exactly equal counts
            as 0.5 (no preference). `nan` if no pairs survive the tie filter.
        n_pairs_used: number of kept (non-tied) pairs the win rate is over.
        n_pairs_tied: number of pairs thrown out as ground-truth ties, counted
            over ALL pairs, not just the subsample.
    """
    scores = np.asarray(scores, dtype=float)
    values = np.asarray(values, dtype=float)
    iu, ju = np.triu_indices(len(scores), k=1)   # all i < j pairs

    # Drop ties before subsampling, so the subsample is n_pairs decidable comparisons.
    tied = np.abs(scores[iu] - scores[ju]) <= tie_tol
    n_tied = int(tied.sum())
    iu, ju = iu[~tied], ju[~tied]
    n_used = len(iu)
    if n_used == 0:
        return float('nan'), 0, n_tied

    if n_pairs is not None and n_pairs < n_used:
        # Drawn once from a fixed seed, so every policy is scored on the SAME pairs.
        # Resampling per call would make a training curve jitter from the pair draw alone.
        sel = np.random.default_rng(seed).choice(n_used, size=n_pairs, replace=False)
        iu, ju = iu[sel], ju[sel]
        n_used = n_pairs

    sd, vd = scores[iu] - scores[ju], values[iu] - values[ju]
    # concordant = model orders the pair the same way as the ground truth.
    # A model tie (vd == 0) is neither concordant nor discordant -> 0.5.
    wins = np.where(vd == 0.0, 0.5, (np.sign(sd) == np.sign(vd)).astype(float))
    return float(wins.mean()), n_used, n_tied
