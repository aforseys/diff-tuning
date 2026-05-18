"""
Visualize saved EEF trajectories from a demo HDF5 file.

Usage:
    conda run -n diffpreff python scripts/visualize_demos.py --demo-file data/robosuite/demos.hdf5
    conda run -n diffpreff python scripts/visualize_demos.py --demo-file data/robosuite/demos.hdf5 --demo 3
    conda run -n diffpreff python scripts/visualize_demos.py --demo-file data/robosuite/demos.hdf5 --all
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_demo(ax_xyz, ax_3d, eef_pos, demo_name, success):
    t = np.arange(len(eef_pos))
    label = f"{demo_name} ({'ok' if success else 'fail'})"

    ax_xyz.plot(t, eef_pos[:, 0], label=f"x {label}")
    ax_xyz.plot(t, eef_pos[:, 1], label=f"y {label}")
    ax_xyz.plot(t, eef_pos[:, 2], label=f"z {label}")

    ax_3d.plot(eef_pos[:, 0], eef_pos[:, 1], eef_pos[:, 2], alpha=0.7, label=label)
    ax_3d.scatter(*eef_pos[0],  color="green", s=40, zorder=5)   # start
    ax_3d.scatter(*eef_pos[-1], color="red",   s=40, zorder=5)   # end


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo-file", required=True, help="HDF5 demo file")
    parser.add_argument("--demo",      type=int, default=0,
                        help="Demo index to visualize (default 0)")
    parser.add_argument("--all",       action="store_true",
                        help="Overlay all demos")
    args = parser.parse_args()

    with h5py.File(args.demo_file, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))

        if args.all:
            selected = demo_keys
        else:
            selected = [demo_keys[args.demo]]

        fig = plt.figure(figsize=(14, 5))
        ax_xyz = fig.add_subplot(1, 2, 1)
        ax_3d  = fig.add_subplot(1, 2, 2, projection="3d")

        for key in selected:
            eef_pos = data[key]["obs/eef_pos"][:]          # (T+1, 3)
            success = bool(data[key].attrs["success"])
            plot_demo(ax_xyz, ax_3d, eef_pos, key, success)

            print(f"{key}  steps={len(eef_pos)}  "
                  f"bin={data[key].attrs['bin_idx']}  "
                  f"prism={data[key].attrs['prism_idx']}  "
                  f"{'SUCCESS' if success else 'FAIL'}")

    ax_xyz.set_xlabel("step")
    ax_xyz.set_ylabel("position (m)")
    ax_xyz.set_title("EEF XYZ over time")
    ax_xyz.legend(fontsize=6)

    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_title("EEF 3D path  (green=start, red=end)")
    ax_3d.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig("demo_eef_viz.png", dpi=150)
    print("\nSaved → demo_eef_viz.png")
    plt.show()


if __name__ == "__main__":
    main()
