"""
Environment-agnostic preference-ranking math.

Nothing here knows about mazes, GMM clusters, or robosuite bins — it just
compares two per-sample arrays (a preference metric and some other value,
e.g. model energy) and reports how well their rankings agree. Environment-
specific scoring functions (e.g. `maze_scoring.py`) produce the `scores`
array; this module only does the correlation.

`rank_correlation` and `pairwise_win_rate` are pure array math. The three below
them -- `dpo_reward`, `policy_energy` and `policy_win_rate` -- do touch a policy,
but only to produce the values being ranked; that computation is identical in
every environment, so each env's eval supplies just the observations that go with
its trajectories and the ground-truth scores to rank against.
"""

import numpy as np
import torch
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


def pairwise_win_rate(scores, values, tie_tol=0.0, n_pairs=None, seed=0, pairs=None):
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
    pairs: (M, 2) array of index pairs to compare, instead of every i < j. Use this
        when the comparisons are already paired -- e.g. maze preference pairs, whose
        two members share an observation, so only those M comparisons are well posed.

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
    if pairs is None:
        iu, ju = np.triu_indices(len(scores), k=1)   # all i < j pairs
    else:
        pairs = np.asarray(pairs)
        iu, ju = pairs[:, 0], pairs[:, 1]

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


def dpo_reward(policy, trajs, observations, ref_policy=None, n_timesteps=10,
               seed=0, batch_size=256):
    """
    DPO's implicit reward over `trajs` (higher = more preferred), one list entry per
    entry in `observations`: -mse for forward-KL DPO, or -(mse - mse_ref) with
    `ref_policy` (traditional DPO), where mse is the denoising error averaged over
    `n_timesteps` uniformly drawn (t, eps).

    The draws are seeded and shared by every trajectory and by both policies, so the
    reward differences between trajectories don't depend on which draws came up.

    trajs: (N, horizon, action_dim) tensor of candidate trajectories.
    observations: list of observation batches, each a dict of (N, ...) tensors covering
        all N trajectories. One entry per conditioning context the caller wants scored
        (GMM sweeps its cluster contexts; maze passes the observations its test
        trajectories were collected from).

    Mean denoising error alone -- this with `ref_policy=None`, negated -- is also the
    "D-MSE" metric, so pass no reference when that is what you want.
    """
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    n_train_timesteps = policy.diffusion.noise_scheduler.config.num_train_timesteps
    timesteps = torch.randint(0, n_train_timesteps, (n_timesteps,), device=device, generator=generator)
    noise = torch.randn((n_timesteps, *trajs.shape[1:]), device=device, generator=generator)

    def denoising_error(p, traj, obs):
        batch = {**p.normalize_inputs(obs), **p.normalize_targets({'action': traj})}
        errors = [p.diffusion._compute_denoising_sq_error(batch, t.expand(len(traj)), eps.expand_as(traj))
                  for t, eps in zip(timesteps, noise)]
        return torch.stack(errors).mean(dim=0)  # (B, 1)

    rewards = []
    for obs in observations:
        outputs = []
        for i in range(0, trajs.size(0), batch_size):
            batch_traj = trajs[i:i+batch_size]
            batch_obs = {k: v[i:i+batch_size] for k, v in obs.items()}
            mse = denoising_error(policy, batch_traj, batch_obs)
            if ref_policy is not None:
                mse = mse - denoising_error(ref_policy, batch_traj, batch_obs)
            outputs.append(-mse.detach().cpu().numpy())
        rewards.append(np.concatenate(outputs, axis=0))
    return rewards


def policy_energy(policy, trajs, observations, t=0, batch_size=256, deterministic=True,
                  **energy_kwargs):
    """
    The policy's energy over `trajs` (lower = more preferred), one list entry per entry
    in `observations`. Energy-based policies only -- plain diffusion has no `get_energy`.

    deterministic defaults True here, unlike `get_energy`, so every environment's eval
    scores energies the same way: no noise added, the trajectory only rescaled to the
    timestep. The remaining `energy_kwargs` (n_noise, seed, mask) pass straight through
    and keep `get_energy`'s defaults -- n_noise is ignored while deterministic is True.
    """
    energies = []
    for obs in observations:
        outputs = []
        for i in range(0, trajs.size(0), batch_size):
            batch_traj = {'action': trajs[i:i+batch_size]}
            batch_obs = {k: v[i:i+batch_size] for k, v in obs.items()}
            out = policy.get_energy(action_batch=batch_traj, t=t, observation_batch=batch_obs,
                                    deterministic=deterministic, **energy_kwargs)
            outputs.append(out.detach().cpu().numpy())
        energies.append(np.concatenate(outputs, axis=0))
    return energies


def policy_win_rate(policy, trajs, observations, gt_scores, score="energy", ref_policy=None,
                    t=0, seed=0, tie_tol=0.0, n_pairs=None, pair_seed=0, pairs=None,
                    **energy_kwargs):
    """
    How often does the policy order a pair the same way `gt_scores` does? One win rate
    per entry in `observations`.

    This is the orchestration around `pairwise_win_rate`: it runs the model to produce
    the values being ranked, then defers all the comparison logic (tie filter,
    subsampling, pairing) to that function.

    score: what the policy ranks by. "energy" uses `policy_energy` (lower = preferred, so
        it is negated); "dpo" uses `dpo_reward`, relative to `ref_policy` when given.
    gt_scores: (N,) ground-truth preference, higher = preferred. Each environment supplies
        this its own way -- a utility for GMM, `maze_scoring.score_metric` for maze.
    pairs: forwarded to `pairwise_win_rate`; pass pre-formed pairs when the comparisons
        are already paired (e.g. two trajectories sharing one observation).
    """
    if score == "energy":
        values = [-e for e in policy_energy(policy, trajs, observations, t=t, seed=seed,
                                            **energy_kwargs)]
    elif score == "dpo":
        values = dpo_reward(policy, trajs, observations, ref_policy=ref_policy, seed=seed)
    else:
        raise ValueError(f"Unknown win rate score '{score}'. Expected 'energy' or 'dpo'.")
    return [pairwise_win_rate(gt_scores, v.reshape(-1), tie_tol=tie_tol,
                              n_pairs=n_pairs, seed=pair_seed, pairs=pairs)[0] for v in values]
