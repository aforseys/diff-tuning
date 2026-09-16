#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""The maze2d environment: trajectories are scored offline against the maze metrics, with no simulator."""
from itps.envs.base_env import Environment
from itps.envs.maze.eval import eval_maze


class MazeEnv(Environment):
    name = "maze2d"

    def evaluate(self, policy, cfg, seed=None, **kwargs):
        """
        Sample trajectories for every saved start observation in `eval.train_obs` / `eval.test_obs` and
        score them with the metrics named in `eval.metrics`.

        `eval_maze` reports each metric per sampling variant; they are flattened here into the single
        `aggregated` dict that training and eval log, as `<split>_<metric>_<variant>`.
        """
        info = {"aggregated": {}}
        if cfg.eval.get("train_obs") is None and cfg.eval.get("test_obs") is None:
            return info
        for split in ("train", "test"):
            split_info = eval_maze(policy, cfg, split=split, seed=seed)
            for label, metrics in split_info.items():
                for metric_name, vals in metrics.items():
                    info["aggregated"][f"{metric_name}_{label}"] = vals["mean"]
        return info
