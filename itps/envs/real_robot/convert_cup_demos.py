"""Convert cup_multi_shelf recordings (SimpleTrajectoryRecorder .h5, one file per demo) into the single hdf5 the
`real_robot` dataset loader reads.

    python itps/envs/real_robot/convert_cup_demos.py --out itps/data/real_robot/cup_multi_shelf.hdf5
    python itps/envs/real_robot/convert_cup_demos.py --out itps/data/real_robot/cup_overfit10.hdf5 \
        --start s1_disc5 --end e1 --max_demos 10

Per demo:
  * only files with attrs['complete'] == True;
  * 10 Hz grid: for each tick t0 + 0.1 n keep the nearest recorded frame. The recorder runs a steady 10 Hz clock
    but also snapshots ~4 extra frames per demo at events (grasp/release start, arrival poses, `middle`, final
    frame), ~40-60 ms off the clock; those are dropped so every step spans 100 ms. A demo with a tick that has no
    frame within GRID_TOL_S (a capture stall, not an event frame) is skipped and reported -- 3 of 300;
  * images through image_pipeline (true RGB despite the recorder's channel quirk, 4:3 crop, 120x160);
  * measured joint positions and gripper width at the kept frames. Action labels are NOT stored: the loader builds
    them per window from these measured values (chunk-wise joint offsets, absolute width).

Output layout (mirrors the robosuite demo files):
    data/demo_<i>/obs/joint_pos        (T, 7)        float64  measured
    data/demo_<i>/obs/gripper_width    (T,)          float64  measured
    data/demo_<i>/obs/<cam>_image      (T, H, W, 3)  uint8    true RGB, policy resolution
    data/demo_<i>/phase                (T,)          int8     recorder phase index (attrs phase_names)
    data/demo_<i>/timestamps           (T,)          float64  recorded times of the kept frames
    data/demo_<i>/source_frame_index   (T,)          int64    index into the source file's frames
    data/demo_<i>.attrs: source_file, demo_id, start, end, n_frames_dropped
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_pipeline import POLICY_HW, decode_recorded, to_policy_image  # noqa: E402

FPS = 10
GRID_TOL_S = 0.03  # on-clock frames are within 20 ms of their tick; event frames are 40-60 ms off
DEFAULT_DEMOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/cup_multi_shelf/demos")


def grid_indices(t: np.ndarray) -> np.ndarray:
    """Indices of the frames nearest each 10 Hz tick from the first frame to the last."""
    rel = t - t[0]
    n_ticks = int(np.floor(rel[-1] * FPS + 0.5)) + 1
    ticks = np.arange(n_ticks) / FPS
    idx = np.abs(rel[None, :] - ticks[:, None]).argmin(axis=1)
    err = np.abs(rel[idx] - ticks)
    # The last tick may fall past the final on-clock frame; drop it rather than fail.
    if err[-1] > GRID_TOL_S:
        idx, err = idx[:-1], err[:-1]
    if (err > GRID_TOL_S).any():
        bad = int(np.argmax(err > GRID_TOL_S))
        raise ValueError(f"no frame within {GRID_TOL_S}s of tick {bad} ({ticks[bad]:.1f}s): gap in recording")
    if (np.diff(idx) <= 0).any():
        raise ValueError("two ticks mapped to the same frame")
    return idx


def convert_one(args):
    path, cameras = args
    with h5py.File(path, "r") as f:
        idx = grid_indices(f["timestamps"][:])
        out = {
            "joint_pos": f["proprioception/joint_positions"][:][idx],
            "gripper_width": f["proprioception/gripper_width"][:][idx],
            "phase": f["phase"][:][idx].astype(np.int8),
            "timestamps": f["timestamps"][:][idx],
            "source_frame_index": idx.astype(np.int64),
            "images": {
                cam: np.stack([to_policy_image(decode_recorded(f[f"images/{cam}"][i])) for i in idx])
                for cam in cameras
            },
            "attrs": {
                "source_file": os.path.basename(path),
                "demo_id": f.attrs["demo_id"],
                "start": f.attrs["start"],
                "end": f.attrs["end"],
                "n_frames_dropped": int(len(f["timestamps"]) - len(idx)),
            },
            "phase_names": f.attrs["phase_names"],
        }
    return out


def select_files(demos_dir, start, end, max_demos):
    """Complete demos matching start/end, minus those whose timing can't be put on the 10 Hz grid.

    Returns (files, skipped) with skipped = [(path, reason)]. Skipping is deliberate: 3 of the 300 complete
    cup_multi_shelf demos have a capture stall (frames 30-80 ms off the clock for a tick or a short burst), and
    dropping them keeps every kept image paired with the joint reading captured with it.
    """
    pattern = f"*_{start or '*'}__{end or '*'}__r*.h5"
    files, skipped = [], []
    for path in sorted(glob.glob(os.path.join(demos_dir, pattern))):
        with h5py.File(path, "r") as f:
            # Filter on the attrs, not just the filename the glob matched.
            if not f.attrs["complete"]:
                continue
            if (start and f.attrs["start"] != start) or (end and f.attrs["end"] != end):
                continue
            try:
                grid_indices(f["timestamps"][:])
            except ValueError as e:
                skipped.append((path, str(e)))
                continue
        files.append(path)
    return (files[:max_demos] if max_demos else files), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos_dir", default=DEFAULT_DEMOS_DIR)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cameras", nargs="+", default=["wrist", "agent"])
    parser.add_argument("--start", default=None, help="only demos with this start, e.g. s1_disc5")
    parser.add_argument("--end", default=None, help="only demos with this end (shelf), e.g. e1")
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    files, skipped = select_files(args.demos_dir, args.start, args.end, args.max_demos)
    for path, reason in skipped:
        print(f"skipping {os.path.basename(path)}: {reason}")
    if not files:
        raise SystemExit("no complete demos matched")
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"converting {len(files)} demos, cameras {args.cameras} -> {args.out}")

    n_frames, n_dropped, phase_names = 0, 0, None
    with h5py.File(args.out, "w") as out, Pool(args.workers) as pool:
        out.attrs["source_dir"] = os.path.abspath(args.demos_dir)
        out.attrs["cameras"] = ",".join(args.cameras)
        out.attrs["image_hw"] = POLICY_HW
        out.attrs["fps"] = FPS
        out.attrs["image_format"] = "RGB (true colour; decoded with image_pipeline.decode_recorded)"
        data = out.create_group("data")
        for i, demo in enumerate(pool.imap(convert_one, [(p, args.cameras) for p in files])):
            if phase_names is None:
                phase_names = demo["phase_names"]
            assert demo["phase_names"] == phase_names, "phase names differ between demos"
            g = data.create_group(f"demo_{i}")
            g.create_dataset("obs/joint_pos", data=demo["joint_pos"])
            g.create_dataset("obs/gripper_width", data=demo["gripper_width"])
            for cam, imgs in demo["images"].items():
                g.create_dataset(f"obs/{cam}_image", data=imgs, chunks=(1, *imgs.shape[1:]),
                                 compression="gzip", compression_opts=4)
            for key in ("phase", "timestamps", "source_frame_index"):
                g.create_dataset(key, data=demo[key])
            for k, v in demo["attrs"].items():
                g.attrs[k] = v
            n_frames += len(demo["timestamps"])
            n_dropped += demo["attrs"]["n_frames_dropped"]
        out.attrs["phase_names"] = phase_names
    print(f"done: {len(files)} demos, {n_frames} frames kept, {n_dropped} off-grid frames dropped")


if __name__ == "__main__":
    main()
