#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""The GMM environment: a 2D Gaussian-mixture action distribution with a known ground-truth density."""
import numpy as np

from itps.envs.base_env import Environment
from itps.envs.gmm.eval import eval_GMM
from itps.envs.gmm.gaussian_mm import DEFAULT_SPEC, get_spec, get_utility


class GMMEnv(Environment):
    name = "gmm"

    def evaluate(self, policy, cfg, seed=None, viz=False, viz_opt=False,
                 training_samples=None, save_samples_path=None, **kwargs):
        """
        Sample from the policy and compare against the ground-truth mixture (KL divergence, where the
        policy exposes energies, and log-likelihood of its samples).

        The ground-truth mixture is whichever spec `env.gmm_spec` names, and carries no
        preference. If `eval.utility` names one, the samples are additionally scored by
        that utility, and by win rate against `eval.pref_test_set` if one is given.
        """
        spec = get_spec(cfg.env.get("gmm_spec", DEFAULT_SPEC))

        utility_name = cfg.eval.get("utility")
        utility = get_utility(utility_name) if utility_name else None

        pref_test_set = cfg.eval.get("pref_test_set")
        # Leading column is the observation.
        pref_test_points = np.load(pref_test_set)[:, 1:] if pref_test_set else None

        return eval_GMM(
            policy,
            spec,
            condition_type=cfg.condition_type.lower(),
            N=cfg.eval.n_samples,
            viz=viz,
            training_samples=training_samples,
            opt_params=list(cfg.eval.opt_params) if cfg.eval.get("opt_params") else None,
            methods=list(cfg.eval.methods),
            viz_opt=viz_opt,
            save_samples_path=save_samples_path,
            seed=seed,
            utility=utility,
            pref_test_points=pref_test_points,
        )
