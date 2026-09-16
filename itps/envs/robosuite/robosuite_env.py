#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""The robosuite bin-placing environment: the policy is rolled out in MuJoCo."""
from itps.envs.base_env import Environment


class RobosuiteEnv(Environment):
    name = "robosuite"

    def evaluate(self, policy, cfg, seed=None, render=False, n_viz_samples=0, **kwargs):
        """
        Roll the policy out from the saved start states in `eval.obs_file` and report placement success
        and trajectory features, split by the seen/new prism and bin conditions.
        """
        # Imported here so the other environments don't need robosuite/MuJoCo installed.
        from itps.envs.robosuite.eval import eval_robosuite

        return eval_robosuite(policy, cfg, seed=seed, render=render, n_viz_samples=n_viz_samples)
