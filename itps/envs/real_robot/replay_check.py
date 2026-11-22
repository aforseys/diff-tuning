"""Offline replay check (ROLLOUT_PLAN B): run the policy on the exact training inputs, before any robot motion.

    python itps/envs/real_robot/replay_check.py <pretrained_model_dir> --hdf5 <cup_overfit10.hdf5> [--stride 1]

For each demo in the hdf5 and each anchor t, frames (t-1, t) go through PolicyRunner exactly as they will live,
and the 8 predicted targets are compared with the recorded measured q[t+1..t+8], w[t+1..t+8].

Reports:
  * per-joint RMSE for each chunk index i, next to the demo-to-demo spread at the same phase: for each (demo, t),
    how far the OTHER demos' recorded chunk (q[t'+i] - q[t'], at the same point of the same recorder phase)
    is from this demo's -- what predicting "some other demo" would score;
  * gripper close/open timing error in frames: executed widths (a chunk every exec_steps frames, as the robot
    runs it) through execution.GripperHysteresis, vs the same hysteresis on the recorded widths;
  * inference latency mean / p95.
Plot: predicted (chunks stitched as executed) vs recorded joint traces for one demo.
Pass: errors comparable to (below) the spread, gripper timing within ~1-2 frames.

Outputs go to <hdf5 dir>/replay_check/<run name>/ (report.txt, errors.npz, traces_demo<k>.png).
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from execution import DEFAULT_EXEC_STEPS, N_JOINTS, hysteresis_events  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402


def load_hdf5_episodes(path, cameras):
    """The converted training file -> [{q (T,7), w (T,), images {cam: (T,H,W,3)}, phase (T,), name}]."""
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"], key=lambda k: int(k.split("_")[1]))
        phase_names = str(f.attrs["phase_names"]).split(",")
        episodes = [{
            "name": f"{k} ({f['data'][k].attrs['source_file']})",
            "q": f[f"data/{k}/obs/joint_pos"][:],
            "w": f[f"data/{k}/obs/gripper_width"][:],
            "phase": f[f"data/{k}/phase"][:],
            "images": {cam: f[f"data/{k}/obs/{cam}_image"][:] for cam in cameras},
        } for k in keys]
    return episodes, phase_names


def predict_episode(runner, ep, anchors):
    """{t: targets (8, 8)} and the latencies, feeding frames (t-1, t) through PolicyRunner as live."""
    preds, latencies = {}, []
    dt = 1.0 / runner.fps
    for t in anchors:
        runner.reset()
        for s in ([t - 1, t] if t > 0 else [0]):
            runner.push({cam: ep["images"][cam][s] for cam in runner.cameras},
                        np.concatenate([ep["q"][s], [ep["w"][s]]]), s * dt)
        targets, _, latency = runner.infer()
        preds[t] = targets
        latencies.append(latency)
    return preds, latencies


def phase_progress(phase):
    """Per frame: segment ordinal + fraction through that segment (segments = runs of one recorder phase)."""
    change = np.nonzero(np.diff(phase))[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(phase)]])
    progress = np.empty(len(phase))
    for k, (a, b) in enumerate(zip(starts, ends)):
        progress[a:b] = k + (np.arange(b - a) / (b - a))
    return progress, tuple(int(phase[a]) for a in starts)


def chunk_rel(q, t, n):
    """Recorded chunk relative to its anchor: q[t+1..t+n] - q[t], NaN past the end."""
    out = np.full((n, q.shape[1]), np.nan)
    k = np.arange(t + 1, min(t + 1 + n, len(q)))
    out[: len(k)] = q[k] - q[t]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pretrained_model_dir")
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--stride", type=int, default=1, help="evaluate every stride-th anchor")
    parser.add_argument("--exec_steps", type=int, default=DEFAULT_EXEC_STEPS,
                        help="re-plan every this many frames, as the robot does (default: execution.DEFAULT_EXEC_STEPS)")
    parser.add_argument("--plot_demo", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runner = PolicyRunner(args.pretrained_model_dir, device=args.device, seed=args.seed)
    n = runner.n_action_steps
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.hdf5)), "replay_check", runner.name)
    os.makedirs(out_dir, exist_ok=True)
    episodes, phase_names = load_hdf5_episodes(args.hdf5, runner.cameras)
    lines = []

    def report(s=""):
        print(s)
        lines.append(s)

    report(f"{runner.name} ({runner.policy_type}) on {runner.device}; {len(episodes)} demos from {args.hdf5}; "
           f"stride {args.stride}, exec_steps {args.exec_steps}")

    # ── predictions ─────────────────────────────────────────────────────────
    all_preds, latencies = [], []
    for d, ep in enumerate(episodes):
        T = len(ep["q"])
        anchors = sorted(set(range(0, T - 1, args.stride)) | set(range(0, T - 1, args.exec_steps)))
        preds, lat = predict_episode(runner, ep, anchors)
        all_preds.append(preds)
        latencies += lat
        print(f"  demo {d}: {len(anchors)} anchors, {T} frames", flush=True)

    # ── joint errors vs demo-to-demo spread ─────────────────────────────────
    progress = [phase_progress(ep["phase"]) for ep in episodes]
    same_phases = len({p[1] for p in progress}) == 1
    if not same_phases:
        report("  [align] demos have different phase sequences -- aligning by normalized time instead")
        progress = [(np.linspace(0, 1, len(ep["q"])), None) for ep in episodes]
    err, spread = [], []  # each (n, 8) per (demo, anchor); NaN where the chunk runs past the end
    for d, ep in enumerate(episodes):
        rec = np.concatenate([ep["q"], ep["w"][:, None]], axis=1)
        for t, targets in all_preds[d].items():
            if t % args.stride:
                continue
            e = np.full((n, rec.shape[1]), np.nan)
            k = np.arange(t + 1, min(t + 1 + n, len(rec)))
            e[: len(k)] = targets[: len(k)] - rec[k]
            err.append(e)
            mine = chunk_rel(rec, t, n)
            others = []
            for d2, ep2 in enumerate(episodes):
                if d2 == d:
                    continue
                t2 = int(np.argmin(np.abs(progress[d2][0] - progress[d][0][t])))
                rec2 = np.concatenate([ep2["q"], ep2["w"][:, None]], axis=1)
                other = chunk_rel(rec2, t2, n)
                other[:, N_JOINTS] = rec2[np.clip(t2 + 1 + np.arange(n), 0, len(rec2) - 1), N_JOINTS]
                others.append(other)
            mine[:, N_JOINTS] = rec[np.clip(t + 1 + np.arange(n), 0, len(rec) - 1), N_JOINTS]
            spread.append(np.sqrt(np.nanmean((np.stack(others) - mine) ** 2, axis=0)))
    err, spread = np.stack(err), np.stack(spread)
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))           # (n, 8)
    spread_rms = np.sqrt(np.nanmean(spread ** 2, axis=0))  # (n, 8)

    report(f"\nper-joint RMSE [mrad] by chunk index i (target t+i), model / demo-to-demo spread at the same phase:")
    report("   i  " + "".join(f"{f'j{j}':>16}" for j in range(N_JOINTS)) + f"{'width [mm]':>16}")
    for i in range(n):
        cells = [f"{rmse[i, j] * 1e3:7.1f}/{spread_rms[i, j] * 1e3:<7.1f}" for j in range(N_JOINTS)]
        cells.append(f"{rmse[i, N_JOINTS] * 1e3:7.1f}/{spread_rms[i, N_JOINTS] * 1e3:<7.1f}")
        report(f"  {i + 1:>2}  " + "".join(f"{c:>16}" for c in cells))
    ratio = rmse[:, :N_JOINTS] / np.maximum(spread_rms[:, :N_JOINTS], 1e-9)
    report(f"  model/spread over joints: median {np.median(ratio):.2f}, worst {ratio.max():.2f} "
           f"(j{np.unravel_index(ratio.argmax(), ratio.shape)[1]}, i={np.unravel_index(ratio.argmax(), ratio.shape)[0] + 1})")

    # ── gripper timing, executed as the robot will ──────────────────────────
    report(f"\ngripper events (hysteresis), executed chunks every {args.exec_steps} frames vs recorded "
           f"[frame index; error = executed - recorded]:")
    timing_errors = []
    executed = []
    for d, ep in enumerate(episodes):
        T = len(ep["q"])
        ex = np.full((T, N_JOINTS + 1), np.nan)
        ex[0] = np.concatenate([ep["q"][0], [ep["w"][0]]])
        for t in range(0, T - 1, args.exec_steps):
            k = np.arange(t + 1, min(t + 1 + args.exec_steps, T))
            ex[k] = all_preds[d][t][: len(k)]
        executed.append(ex)
        rec_ev, ex_ev = hysteresis_events(ep["w"]), hysteresis_events(ex[:, N_JOINTS])
        cells = []
        for kind in ("close", "open"):
            r = [i for i, k in rec_ev if k == kind]
            x = [i for i, k in ex_ev if k == kind]
            if r and x:
                timing_errors.append(x[0] - r[0])
                cells.append(f"{kind} {x[0]} vs {r[0]} ({x[0] - r[0]:+d})")
            else:
                timing_errors.append(np.nan)
                cells.append(f"{kind} executed {x[:1] or 'never'} vs recorded {r[:1] or 'never'}")
        extra = "" if (len(rec_ev), len(ex_ev)) == (2, 2) else f"  [events: recorded {rec_ev}, executed {ex_ev}]"
        report(f"  demo {d}: " + ", ".join(cells) + extra)
    te = np.array(timing_errors, dtype=float)
    report(f"  worst |timing error| {np.nanmax(np.abs(te)):.0f} frames; missing events: {int(np.isnan(te).sum())}")

    lat = np.array(latencies)
    report(f"\nlatency: mean {lat.mean() * 1000:.1f} ms, p95 {np.percentile(lat, 95) * 1000:.1f} ms, "
           f"max {lat.max() * 1000:.1f} ms over {len(lat)} calls")

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(os.path.join(out_dir, "errors.npz"), err=err, spread=spread, rmse=rmse, spread_rms=spread_rms,
             timing_errors=te, latencies=lat)

    # ── plot: one demo, executed vs recorded ────────────────────────────────
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = args.plot_demo
    ep, ex = episodes[d], executed[d]
    t_axis = np.arange(len(ep["q"])) / runner.fps
    fig, axes = plt.subplots(N_JOINTS + 1, 1, figsize=(10, 14), sharex=True)
    rec = np.concatenate([ep["q"], ep["w"][:, None]], axis=1)
    for j, ax in enumerate(axes):
        ax.plot(t_axis, rec[:, j], color="0.3", lw=1.5, label="recorded")
        ax.plot(t_axis, ex[:, j], color="C1", lw=1.0, marker=".", ms=3, label="predicted (executed chunks)")
        for t in range(0, len(t_axis), args.exec_steps):
            ax.axvline(t / runner.fps, color="0.9", lw=0.5, zorder=0)
        ax.set_ylabel(f"j{j} [rad]" if j < N_JOINTS else "width [m]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{runner.name}: {ep['name']}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"traces_demo{d}.png"), dpi=110)
    print(f"\nwrote {out_dir}/report.txt, errors.npz, traces_demo{d}.png")


if __name__ == "__main__":
    main()
