#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Environment interface shared by the research domains (GMM, maze2d, robosuite).

An Environment bundles everything domain-specific about a policy's evaluation: how to sample from the
policy, what to measure, and how to report it. `train.py` and `eval.py` look one up by `cfg.env.name`
(see `itps.envs.get_env`) instead of branching per domain, and gym environments that LeRobot evaluates
by rollout stay on its own `eval_policy` path.
"""
from abc import ABC, abstractmethod


class Environment(ABC):
    """One research domain's evaluation."""

    name: str

    @abstractmethod
    def evaluate(self, policy, cfg, seed: int | None = None, **kwargs) -> dict:
        """
        Evaluate `policy` as `cfg.eval` specifies.

        Returns {"aggregated": {metric_name: value}}; the aggregated dict is what training and eval log,
        so its keys are the metric names that end up in wandb.
        """
