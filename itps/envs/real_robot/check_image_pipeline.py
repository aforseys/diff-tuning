"""Check that training images (recorder -> .h5 -> conversion) and live rollout images go through identical
processing. Run from anywhere:

    python itps/envs/real_robot/check_image_pipeline.py [--recorder_src ~/Desktop/SimpleTrajectoryRecorder/src]

Uses the recorder's ACTUAL encode_frame_jpeg, so a change to the recorder's image path shows up here.
  1. colour: the cup decodes red (true RGB) through decode_recorded;
  2. exactness: for the same camera frame, conversion path == live_frame(jpeg_roundtrip=True) path, bit for bit;
  3. reports how far the default live path (no JPEG round trip) is from the training pixels.
"""
import argparse
import glob
import os
import sys
import types

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_pipeline import RECORDER_JPEG_QUALITY, decode_recorded, live_frame, to_policy_image  # noqa: E402

DEMOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/cup_multi_shelf/demos")


def import_recorder_encode(recorder_src):
    """Load the recorder's episodes.py on its own: the simple_recorder package __init__ pulls in robot drivers
    (zero_franky, ...), and episodes.py imports threed_mouse for an unrelated helper, which is stubbed here."""
    import importlib.util

    sys.modules.setdefault("threed_mouse", types.ModuleType("threed_mouse"))
    geom = sys.modules.setdefault("threed_mouse.geometry", types.ModuleType("threed_mouse.geometry"))
    geom.R_to_rot_vector = getattr(geom, "R_to_rot_vector", lambda *a, **k: None)
    path = os.path.join(os.path.expanduser(recorder_src), "simple_recorder", "episodes.py")
    spec = importlib.util.spec_from_file_location("recorder_episodes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_frame_jpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder_src", default="~/Desktop/SimpleTrajectoryRecorder/src")
    args = parser.parse_args()
    encode_frame_jpeg = import_recorder_encode(args.recorder_src)

    path = sorted(p for p in glob.glob(os.path.join(DEMOS_DIR, "*.h5")))[0]
    with h5py.File(path, "r") as f:
        frames = {cam: decode_recorded(f[f"images/{cam}"][len(f["timestamps"]) // 3]) for cam in ("wrist", "agent")}

    # 1. colour: strongly saturated pixels (the cup) should be red-dominant.
    for cam, rgb in frames.items():
        a = rgb.astype(int)
        sat = (a.max(-1) - a.min(-1)) > 80
        mean = a[sat].mean(0).round().astype(int)
        print(f"[colour] {cam}: {sat.mean() * 100:.2f}% saturated px, mean RGB {tuple(mean)}")
        assert sat.any() and mean[0] > mean[2], "cup is not red-dominant: decode_recorded channel order is wrong"

    # 2./3. Treat each decoded frame as a raw camera frame and push it through both paths.
    for cam, rgb in frames.items():
        camera_bgr = np.ascontiguousarray(rgb[..., ::-1])  # what cameras.py would return
        # Training path: recorder (flip on capture, encode_frame_jpeg) -> .h5 bytes -> conversion.
        stored = encode_frame_jpeg(np.ascontiguousarray(camera_bgr[..., ::-1]), RECORDER_JPEG_QUALITY)
        train_img = to_policy_image(decode_recorded(stored))
        exact = to_policy_image(live_frame(camera_bgr, jpeg_roundtrip=True))
        default = to_policy_image(live_frame(camera_bgr))
        assert np.array_equal(train_img, exact), f"{cam}: live (jpeg_roundtrip=True) != training pixels"
        d = np.abs(train_img.astype(int) - default.astype(int))
        print(f"[exact] {cam}: jpeg_roundtrip=True matches training bit for bit; "
              f"default live path differs by mean {d.mean():.2f}, p99 {np.percentile(d, 99):.0f}, max {d.max()} "
              f"(uint8 levels)")
    print("OK")


if __name__ == "__main__":
    main()
