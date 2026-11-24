#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""
Agreement between two policies' noise fields at the same query points.

The distillation runs ask whether an energy-parameterized student reproduces a plain diffusion
teacher's eps in the region the denoising loss never supervises. That is a per-timestep question
about directions, not a scalar loss, so it needs cosine similarity and relative error binned by t.
`envs.maze.eval.eval_field_match` (logged during training) and `scripts/eval/compare_fields.py`
(ad-hoc CLI) both call this, so the two cannot report different numbers.
"""
import torch


def field_match_stats(eps_a: torch.Tensor, eps_b: torch.Tensor) -> dict[str, float]:
    """
    How closely eps_a matches eps_b over the batch. Both (B, ...); non-batch dims are flattened.

    cos_mean, cos_p5/p50/p95: cosine similarity per sample, then summarized. The low tail is the
        informative part -- a mean of 0.99 can hide a handful of badly wrong directions, and in
        128-D a cosine of 0.95 is already ~31% transverse error, which is what puts a sampled
        trajectory through a wall.
    rel: ||a - b||^2 / ||b||^2 per sample, averaged -- scale error, which cosine is blind to.
    norm_a, norm_b: mean ||eps||, for comparison against sqrt(D) (what a true N(0, I) draw gives).
    """
    a, b = eps_a.flatten(1).float(), eps_b.flatten(1).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    rel = (a - b).pow(2).sum(1) / b.pow(2).sum(1).clamp_min(1e-12)
    quantiles = torch.quantile(cos, torch.tensor([0.05, 0.5, 0.95], device=cos.device, dtype=cos.dtype))
    return {
        "cos_mean": cos.mean().item(),
        "cos_p5": quantiles[0].item(),
        "cos_p50": quantiles[1].item(),
        "cos_p95": quantiles[2].item(),
        "rel": rel.mean().item(),
        "norm_a": a.norm(dim=1).mean().item(),
        "norm_b": b.norm(dim=1).mean().item(),
    }
