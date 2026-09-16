#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""The GMM environment: a 2D Gaussian-mixture action distribution with a known ground-truth density."""
from omegaconf import DictConfig

from itps.envs.base_env import Environment
from itps.envs.gmm.eval import eval_GMM


class GMMEnv(Environment):
    name = "gmm"

    def evaluate(self, policy, cfg, seed=None, viz=False, viz_opt=False,
                 training_samples=None, save_samples_path=None, **kwargs):
        """
        Sample from the policy and compare against the ground-truth mixture (KL divergence, where the
        policy exposes energies, and log-likelihood of its samples).

        A finetuning run is one given several datasets (`dataset_root` is a dict of base/pos/neg or demo),
        and is scored against the preferred cluster rather than the full mixture.
        """
        return eval_GMM(
            policy,
            condition_type=cfg.condition_type.lower(),
            finetune=isinstance(cfg.dataset_root, DictConfig),
            N=cfg.eval.n_samples,
            viz=viz,
            training_samples=training_samples,
            opt_params=list(cfg.eval.opt_params) if cfg.eval.get("opt_params") else None,
            methods=list(cfg.eval.methods),
            viz_opt=viz_opt,
            save_samples_path=save_samples_path,
            seed=seed,
        )
