#!/usr/bin/env python
"""
generate_configs.py

Reads a joint config YAML file and generates individual run-specific config
YAML files for all combinations of sweep parameters defined under the `sweep:` key.

Run this ONCE before submitting your job array:
    python generate_configs.py --config configs/joint_config.yaml --out_dir configs/runs/

The `sweep:` section of the joint config should mirror the structure of
parameters to vary, with lists of values. For example:

    sweep:
      training:
        lr: [1.0e-4, 1.0e-3]
        batch_size: [256, 512]
      policy:
        energy_landscape_loss_weight: [0.5, 1.0]

All other keys in the joint config are treated as base values and copied
into every generated config file, with sweep parameters overriding them.
Generated configs are saved as: (depending on sweep)
    configs/policy_param_tuning/.../run_0.yaml, configs/policy_param_tuning/.../run_1.yaml, ...

You can inspect these before submitting to verify they look correct.
"""

import argparse
import copy
import itertools
import os

import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def flatten_sweep(sweep_dict, prefix=None):
    """
    Recursively flattens a nested sweep dict into a list of (key_path, [values]) pairs.
    key_path is a list of keys representing the path in the config dict.

    E.g. {"training": {"lr": [1e-4, 1e-3]}}
      -> [(["training", "lr"], [1e-4, 1e-3])]
    """
    if prefix is None:
        prefix = []
    items = []
    for k, v in sweep_dict.items():
        path = prefix + [k]
        if isinstance(v, dict):
            items.extend(flatten_sweep(v, prefix=path))
        elif isinstance(v, list):
            items.append((path, v))
        else:
            # Single value — still include as a fixed override
            items.append((path, [v]))
    return items


def set_nested(d, key_path, value):
    """Sets a value in a nested dict given a list of keys."""
    for key in key_path[:-1]:
        d = d.setdefault(key, {})
    d[key_path[-1]] = value


def generate_configs(joint_config_path, out_dir):
    """
    Reads a joint config YAML, computes all sweep combinations, and writes
    one config YAML per combination to out_dir.

    Returns:
        List[str]: paths to all generated config files
    """
    cfg = load_yaml(joint_config_path)

    sweep = cfg.pop("sweep", {})
    if not sweep:
        raise ValueError("No 'sweep' section found in config. Nothing to sweep over.")

    # Flatten nested sweep dict to [(key_path, [values]), ...]
    flat_params = flatten_sweep(sweep)
    key_paths = [kp for kp, _ in flat_params]
    value_lists = [v for _, v in flat_params]

    # Compute cartesian product of all value lists
    combinations = list(itertools.product(*value_lists))

    os.makedirs(out_dir, exist_ok=True)

    generated_paths = []
    for i, combo in enumerate(combinations):
        # Deep copy base config (sweep section already removed)
        run_cfg = copy.deepcopy(cfg)

        # Apply this combination's values
        for key_path, val in zip(key_paths, combo):
            set_nested(run_cfg, key_path, val)

        # Save to disk
        out_path = os.path.join(out_dir, f"run_{i}.yaml")
        save_yaml(run_cfg, out_path)
        generated_paths.append(out_path)

        # Print summary
        overrides = ", ".join(
            f"{'.'.join(kp)}={val}" for kp, val in zip(key_paths, combo)
        )
        print(f"[{i}] {out_path}  ({overrides})")

    print(f"\nGenerated {len(generated_paths)} config(s) in '{out_dir}'")
    return generated_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to joint config YAML")
    parser.add_argument("--out_dir", default="configs/policy_param_tuning", help="Directory to write run configs")
    args = parser.parse_args()

    generate_configs(args.config, args.out_dir)
