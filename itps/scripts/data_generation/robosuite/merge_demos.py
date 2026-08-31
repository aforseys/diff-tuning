"""
Merge multiple collect_demos.py HDF5 files into one.

Re-indexes demo groups sequentially and adds source_file + strategy_idx
attributes to each demo so its origin is traceable. Useful when demos were
collected separately per strategy and need to be combined for training.

Run from the itps/ directory:
    conda run -n diffpreff python scripts/merge_demos.py \\
        --inputs data/demos_a.hdf5 data/demos_b.hdf5 data/demos_c.hdf5 \\
        --save-path data/merged.hdf5
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs",    required=True, nargs="+",
                        help="Input HDF5 files to merge (in order)")
    parser.add_argument("--save-path", required=True,
                        help="Output HDF5 file path")
    args = parser.parse_args()

    # --- Validate inputs ---
    for path in args.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    # --- Collect metadata from all input files ---
    object_types = []
    config_jsons = []
    demo_counts  = []

    for path in args.inputs:
        with h5py.File(path, "r") as f:
            object_types.append(str(f.attrs.get("object_type", "")))
            config_jsons.append(str(f.attrs.get("config_json", "{}")))
            demo_counts.append(int(f.attrs.get("n_demos", len(f["data"]))))

    unique_objects = set(object_types)
    if len(unique_objects) > 1:
        print(f"WARNING: input files have different object_type values: {unique_objects}")
        print("         Proceeding, but check this is intentional.")

    total_demos = sum(demo_counts)
    print(f"Merging {len(args.inputs)} files → {total_demos} total demos")
    for i, (path, count) in enumerate(zip(args.inputs, demo_counts)):
        print(f"  strategy {i}: {os.path.basename(path)}  ({count} demos)")

    # --- Write merged file ---
    with h5py.File(args.save_path, "w") as out:
        out.attrs["n_demos"]      = total_demos
        out.attrs["object_type"]  = object_types[0]
        out.attrs["config_json"]  = json.dumps(config_jsons)  # list of per-file configs

        data_grp    = out.create_group("data")
        demo_num    = 0
        n_success   = 0

        for strategy_idx, (path, count) in enumerate(zip(args.inputs, demo_counts)):
            source_name = os.path.basename(path)
            with h5py.File(path, "r") as src:
                src_data = src["data"]
                src_keys = sorted(src_data.keys(),
                                  key=lambda k: int(k.split("_")[1]))
                for src_key in src_keys:
                    src_demo = src_data[src_key]
                    dst_key  = f"demo_{demo_num}"
                    dst_demo = data_grp.create_group(dst_key)

                    # Copy all existing attrs
                    for k, v in src_demo.attrs.items():
                        dst_demo.attrs[k] = v

                    # Add provenance attrs
                    dst_demo.attrs["source_file"]   = source_name
                    dst_demo.attrs["strategy_idx"]  = strategy_idx

                    # Copy all datasets
                    for ds_path in src_demo:
                        src_demo.copy(ds_path, dst_demo)

                    if src_demo.attrs.get("success", False):
                        n_success += 1

                    demo_num += 1

            print(f"  strategy {strategy_idx}: copied {count} demos")

    print(f"\nSaved → {args.save_path}")
    print(f"Total: {total_demos} demos  |  success rate: {n_success}/{total_demos} ({100*n_success/total_demos:.1f}%)")


if __name__ == "__main__":
    main()
