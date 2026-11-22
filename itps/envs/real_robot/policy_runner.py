"""The model side of a real-robot rollout: load a checkpoint, keep the observation history, return absolute targets.

Robot-independent on purpose -- it takes policy images (uint8 RGB at the policy resolution, what
image_pipeline.to_policy_image returns and what the converted hdf5 stores), so the offline replay check and the
live rollout share everything after to_policy_image.

    runner = PolicyRunner(".../pretrained_model")
    runner.reset()
    runner.push({"wrist": img, "agent": img}, state, t)      # every 10 Hz observation tick
    targets, times, latency_s = runner.infer()              # (8, 8) absolute [q(7), width], (8,) seconds

Interface contract (ROLLOUT_PLAN.md section 2):
  * observation.state = [measured q (7), measured gripper width (m)]; images keyed by camera name;
  * n_obs_steps = 2 frames exactly 1/fps apart; at an episode start the first frame is duplicated (as training);
  * run_inference(methods=("ddim",)) returns chunk-relative labels for t+1 ... t+n_action_steps;
  * targets = from_chunk_relative(labels, the state that was fed in) -- joints anchored on q_obs[t], width absolute.
    Target i is the expected MEASURED state at t_obs + i / fps.

Benchmark (eval step 1):
    python itps/envs/real_robot/policy_runner.py <pretrained_model_dir> --bench [--n 50]
"""
import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_pipeline import to_model_input  # noqa: E402

from itps.common.datasets.chunk_relative import from_chunk_relative  # noqa: E402
from itps.common.policies.factory import make_policy  # noqa: E402
from itps.common.utils.utils import get_safe_torch_device, init_hydra_config  # noqa: E402

# Consecutive observations must be 1/fps apart within this (the converter's grid tolerance, GRID_TOL_S).
DT_TOL_S = 0.03
IMAGE_KEY_PREFIX = "observation.image."


def checkpoint_name(pretrained_model_dir):
    """A short, unique name for a checkpoint folder (rollout file names, dry-run labels, replay_check folders):

        checkpoints/dp_overfit10_030000/pretrained_model       -> dp_overfit10_030000
        checkpoints/diffusion_visual_capacity                  -> diffusion_visual_capacity
        checkpoints/diffusion_visual/400000/pretrained_model   -> diffusion_visual@400000
        <run>/checkpoints/400000/pretrained_model (Supercloud) -> <run>@400000

    A trailing `pretrained_model` is dropped; a numbered step folder becomes `<run>@<step>`, where <run> skips
    a `checkpoints` folder; anything else is the folder's own name."""
    p = Path(pretrained_model_dir)
    if p.name == "pretrained_model":
        p = p.parent
    if p.name.isdigit():
        run = p.parent.parent if p.parent.name == "checkpoints" else p.parent
        return f"{run.name}@{p.name}"
    return p.name


def check_observation(spec, policy_images, state, t, last_t):
    """Validate one observation against a runner's spec (cameras, image_hw, state_dim, fps): raises ValueError,
    else returns the state as float64 (n,). last_t: the previous observation's time in this episode, or None.
    Shared by PolicyRunner.push and policy_process.PolicyProcess.push (which validates in the parent)."""
    if set(policy_images) != set(spec.cameras):
        raise ValueError(f"expected images for {spec.cameras}, got {sorted(policy_images)}")
    for cam, img in policy_images.items():
        if img.dtype != np.uint8 or img.shape != (*spec.image_hw[cam], 3):
            raise ValueError(f"{cam}: expected uint8 {(*spec.image_hw[cam], 3)}, got {img.dtype} {img.shape}")
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape != (spec.state_dim,):
        raise ValueError(f"expected state ({spec.state_dim},), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"state is not finite: {state}")
    if last_t is not None:
        dt = float(t) - last_t
        if abs(dt - 1.0 / spec.fps) > DT_TOL_S:
            raise ValueError(f"observations {dt * 1000:.0f} ms apart; the policy was trained on "
                             f"{1000 / spec.fps:.0f} ms (+-{DT_TOL_S * 1000:.0f})")
    return state


class PolicyRunner:
    def __init__(self, pretrained_model_dir, device=None, seed=None):
        pm = Path(pretrained_model_dir).expanduser().resolve()
        hydra_cfg = init_hydra_config(str(pm / "config.yaml"))
        if device is not None:
            hydra_cfg.device = device
        elif hydra_cfg.device == "cuda" and not torch.cuda.is_available():
            hydra_cfg.device = "cpu"
        if seed is not None:
            torch.manual_seed(seed)
        self.policy = make_policy(hydra_cfg, pretrained_policy_name_or_path=str(pm))
        self.policy.eval()  # mandatory: training mode applies a random crop
        self.device = get_safe_torch_device(hydra_cfg.device)
        self.name = checkpoint_name(pm)
        self.policy_type = hydra_cfg.policy.name

        cfg = self.policy.config
        self.fps = float(hydra_cfg.fps)
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self.image_keys = [k for k in cfg.input_shapes if k.startswith(IMAGE_KEY_PREFIX)]
        self.cameras = [k[len(IMAGE_KEY_PREFIX):] for k in self.image_keys]
        self.image_hw = {cam: tuple(cfg.input_shapes[k][1:]) for cam, k in zip(self.cameras, self.image_keys)}
        self.state_dim = int(cfg.input_shapes["observation.state"][0])
        self.reset()

    def reset(self):
        """Start a new episode: the next push becomes the first frame (and is duplicated)."""
        self._lock = threading.Lock()  # infer() may run on another thread than push()
        self._history = deque(maxlen=self.n_obs_steps)

    def push(self, policy_images, state, t):
        """Add one observation. policy_images: {camera: uint8 (H, W, 3) RGB}; state: (8,); t: seconds."""
        with self._lock:
            last_t = self._history[-1][2] if self._history else None
        state = check_observation(self, policy_images, state, t, last_t)
        frame = ({cam: np.ascontiguousarray(img) for cam, img in policy_images.items()}, state, float(t))
        with self._lock:
            if self._history:
                self._history.append(frame)
            else:
                self._history.extend([frame] * self.n_obs_steps)  # episode start: duplicate, as training did

    def snapshot(self):
        """The observation history as it is now (what infer() would use)."""
        with self._lock:
            return list(self._history)

    def infer(self, frames=None):
        """-> (targets (n_action_steps, 8) absolute, times (n_action_steps,) seconds, latency_s).
        Uses the history as it is when called (safe to call from another thread than push), or `frames`, a
        snapshot() taken earlier."""
        if frames is None:
            frames = self.snapshot()
        if not frames:
            raise RuntimeError("push an observation before infer()")
        obs = {"observation.state": torch.from_numpy(
            np.stack([s for _, s, _ in frames]).astype(np.float32))[None].to(self.device)}
        for cam, key in zip(self.cameras, self.image_keys):
            obs[key] = torch.from_numpy(np.stack([to_model_input(imgs[cam]) for imgs, _, _ in frames]))[None].to(
                self.device)
        start = time.perf_counter()
        labels = self.policy.run_inference(obs, methods=("ddim",))[0]
        labels = labels[0].detach().cpu().numpy().astype(np.float64)  # .cpu() waits for the GPU
        latency = time.perf_counter() - start
        if labels.shape != (self.n_action_steps, self.state_dim):
            raise RuntimeError(f"policy returned {labels.shape}, expected {(self.n_action_steps, self.state_dim)}")
        anchor_state, t_obs = frames[-1][1], frames[-1][2]
        targets = from_chunk_relative(labels, anchor_state)
        times = t_obs + np.arange(1, self.n_action_steps + 1) / self.fps
        return targets, times, latency


def bench(runner, n):
    rng = np.random.default_rng(0)
    runner.reset()
    for i in range(runner.n_obs_steps):
        images = {cam: rng.integers(0, 256, (*hw, 3), dtype=np.uint8) for cam, hw in runner.image_hw.items()}
        runner.push(images, np.zeros(runner.state_dim), i / runner.fps)
    for _ in range(3):  # warm-up (cudnn autotune, allocator)
        runner.infer()
    latencies = np.array([runner.infer()[2] for _ in range(n)])
    print(f"{runner.name} ({runner.policy_type}) on {runner.device}: latency mean {latencies.mean() * 1000:.1f} ms, "
          f"p95 {np.percentile(latencies, 95) * 1000:.1f} ms, max {latencies.max() * 1000:.1f} ms over {n} calls")
    return latencies


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pretrained_model_dir")
    parser.add_argument("--device", default=None)
    parser.add_argument("--bench", action="store_true", help="time infer() on a dummy observation")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    runner = PolicyRunner(args.pretrained_model_dir, device=args.device)
    print(f"loaded {runner.name}: {runner.policy_type}, cameras {runner.cameras}, images {runner.image_hw}, "
          f"n_obs_steps {runner.n_obs_steps}, n_action_steps {runner.n_action_steps}, fps {runner.fps:g}")
    if args.bench:
        bench(runner, args.n)


if __name__ == "__main__":
    main()
