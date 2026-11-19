"""Chunk-wise relative joint actions (`action_anchor: chunk`): the ONE definition, shared by the dataset loader,
the rollout and the tests.

Every joint action in a window is an offset from a single anchor -- the joint positions in the window's current
observation (the last `observation.state` step):

    label_i  = target_i - q_anchor        (dataset, `to_chunk_relative`)
    target_i = q_anchor + label_i          (rollout, `from_chunk_relative`)

Never a cumulative sum: per-step increments chained together accumulate prediction error along the chunk.
Only the first N_ANCHORED_DIMS action dimensions (the 7 joints) are anchored; the gripper width stays absolute.
"""
import numpy as np
import torch

N_ANCHORED_DIMS = 7


def to_chunk_relative(actions, anchor_state):
    """actions (..., horizon, action_dim) absolute targets, anchor_state (..., state_dim) -> chunk-relative copy."""
    rel = actions.clone() if isinstance(actions, torch.Tensor) else np.array(actions, copy=True)
    rel[..., :N_ANCHORED_DIMS] -= anchor_state[..., None, :N_ANCHORED_DIMS]
    return rel


def from_chunk_relative(rel_actions, anchor_state):
    """Inverse of to_chunk_relative: chunk-relative (..., horizon, action_dim) -> absolute targets."""
    out = rel_actions.clone() if isinstance(rel_actions, torch.Tensor) else np.array(rel_actions, copy=True)
    out[..., :N_ANCHORED_DIMS] += anchor_state[..., None, :N_ANCHORED_DIMS]
    return out


def window_action_stats(actions, states, episode_from, episode_to, action_offsets, drop_n_last_frames):
    """Min/max/mean/std of the chunk-relative action labels over every window the sampler can draw.

    Mirrors training exactly: anchors are the rows EpisodeAwareSampler yields (each episode minus its last
    `drop_n_last_frames` rows), each window covers rows anchor + action_offsets, and entries outside the episode
    (padding, masked out of the loss) are excluded.

    actions, states: (N, action_dim) / (N, state_dim) arrays, rows ordered by episode.
    episode_from, episode_to: per-episode [from, to) row ranges.
    action_offsets: integer row offsets of the action window relative to the anchor (e.g. -1 .. 14).
    """
    offsets = np.asarray(action_offsets)
    chunks = []
    for start, end in zip(episode_from, episode_to):
        a, s = actions[start:end], states[start:end]
        anchors = np.arange(0, (end - start) - drop_n_last_frames)
        rows = anchors[:, None] + offsets[None, :]                       # (n_anchors, horizon)
        valid = (rows >= 0) & (rows < end - start)
        rel = to_chunk_relative(a[np.clip(rows, 0, end - start - 1)], s[anchors])
        chunks.append(rel[valid])
    labels = np.concatenate(chunks, axis=0)
    return {
        "min": torch.from_numpy(labels.min(axis=0)).float(),
        "max": torch.from_numpy(labels.max(axis=0)).float(),
        "mean": torch.from_numpy(labels.mean(axis=0)).float(),
        "std": torch.from_numpy(labels.std(axis=0)).float(),
    }
