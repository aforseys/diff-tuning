#!/usr/bin/env python
"""
run_job.py

Called by the supercloud bash script. Takes LLSUB_RANK and LLSUB_SIZE,
finds all generated config files in the configs directory, assigns a subset
to this process, and runs file.py for each assigned config.

Assumes generate_configs.py has already been run to populate the configs directory.

Usage:
    python run_job.py <LLSUB_RANK> <LLSUB_SIZE> --configs_dir <configs/runs/>
                      --script <file.py> --env <env_name>

Example:
    python run_job.py 0 4 --configs_dir configs/runs/ --script train.py --env gmm
"""

import argparse
import glob
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int, help="LLSUB_RANK: this process's index")
    parser.add_argument("size", type=int, help="LLSUB_SIZE: total number of processes")
    parser.add_argument("--configs_dir", required=True, help="Directory containing generated run config YAMLs")
    parser.add_argument("--script", required=True, help="Path to the python script to run (e.g. train.py)")
    parser.add_argument("--env", required=True, help="Environment name, passed as env={env_name}")
    args = parser.parse_args()

    # Collect all generated config files, sorted so ordering is consistent across all processes
    config_files = sorted(glob.glob(os.path.join(args.configs_dir, "run_*.yaml")))
    total = len(config_files)

    if total == 0:
        raise RuntimeError(
            f"No config files found in '{args.configs_dir}'. "
            "Did you run generate_configs.py first?"
        )

    print(f"[Rank {args.rank}/{args.size}] Total configs: {total}")

    # Slice this process's assigned configs
    # e.g. rank=1, size=4 -> indices 1, 5, 9, ...
    my_configs = config_files[args.rank:total:args.size]

    print(f"[Rank {args.rank}/{args.size}] Assigned {len(my_configs)} config(s):")
    for c in my_configs:
        print(f"  {c}")

    # Run file.py for each assigned config
    for i, config_path in enumerate(my_configs):
        # Pass the config file path as the policy config
        # Hydra will load it via --config-path / --config-name or directly
        full_path = os.path.abspath(config_path)
        relative = full_path.split("policy/", 1)[-1]


        # Extract just the config filename (e.g., "run_config_A")
        config_name = os.path.splitext(os.path.basename(config_path))[0]

        cmd = (
            f"python {args.script} "
            f"env={args.env} "
            f"policy={relative} "
            f"config_name={config_name}"
            )
        
        print(f"\n[Rank {args.rank}/{args.size}] Running combo {i+1}/{len(my_configs)}:")
        print(f"  {cmd}\n")

        result = subprocess.run(cmd, shell=True)

        if result.returncode != 0:
            print(
                f"[Rank {args.rank}/{args.size}] WARNING: Command exited with "
                f"return code {result.returncode} for config: {config_path}"
            )
            # Continue to next config rather than crashing the whole process
            continue

    print(f"\n[Rank {args.rank}/{args.size}] Done.")


if __name__ == "__main__":
    main()
