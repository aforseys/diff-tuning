"""Offline tests for the rollout pieces: execution.py (no model) and PolicyRunner (needs a checkpoint).

    python -m pytest itps/envs/real_robot/test_rollout.py -q
PolicyRunner tests use $ROLLOUT_CKPT (a pretrained_model dir) and are skipped when it isn't there.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from execution import ChunkSchedule, GripperHysteresis, SagCompensation, hysteresis_events  # noqa: E402

CKPT = os.environ.get("ROLLOUT_CKPT",
                      "/home/clear/aforsey/data/cup/checkpoints/dp_overfit10_030000/pretrained_model")


def chunk(t_obs, n=8, w=0.08):
    times = t_obs + np.arange(1, n + 1) / 10
    targets = np.zeros((n, 8))
    targets[:, :7] = np.arange(1, n + 1)[:, None] * np.linspace(0.01, 0.07, 7)
    targets[:, 7] = w
    return targets, times


def test_interpolation_hits_each_target_at_its_time():
    s = ChunkSchedule(exec_steps=8)
    targets, times = chunk(1.0)
    q_from = np.full(7, -0.5)
    assert s.set_chunk(targets, times, tick=10, t_now=1.0, q_from=q_from) == 0
    assert np.allclose(s.joint_target(1.0), q_from)
    for i in range(8):
        assert np.allclose(s.joint_target(times[i]), targets[i, :7])
    mid = 0.5 * (times[2] + times[3])
    assert np.allclose(s.joint_target(mid), 0.5 * (targets[2, :7] + targets[3, :7]))
    assert np.allclose(s.joint_target(5.0), targets[7, :7])  # holds the last target


def test_late_chunk_drops_past_targets_and_starts_from_the_current_command():
    s = ChunkSchedule(exec_steps=8)
    targets, times = chunk(1.0)
    q_cmd = np.full(7, 0.3)
    t_now = times[1] + 0.01  # arrives after targets 1 and 2 were due
    assert s.set_chunk(targets, times, tick=10, t_now=t_now, q_from=q_cmd) == 2
    assert np.allclose(s.joint_target(t_now), q_cmd)
    assert np.allclose(s.joint_target(times[2]), targets[2, :7])
    assert len(s.pending_targets) == 6


def test_all_targets_stream_and_exec_steps_only_sets_the_next_inference():
    s = ChunkSchedule(exec_steps=3)
    assert s.due(0)
    targets, times = chunk(0.0)
    s.set_chunk(targets, times, tick=0, t_now=0.0, q_from=np.zeros(7))
    assert not s.due(2) and s.due(3)
    assert len(s.pending_targets) == 8                        # every target is streamed (F6)
    assert np.allclose(s.joint_target(times[7]), targets[7, :7])
    assert s.held_s == 0.0
    s8 = ChunkSchedule(exec_steps=8)
    s8.set_chunk(targets, times, tick=5, t_now=0.0, q_from=np.zeros(7))
    assert not s8.due(12) and s8.due(13)


def test_held_s_is_the_wait_on_the_last_target():
    s = ChunkSchedule(exec_steps=7)
    targets, times = chunk(0.0)
    s.set_chunk(targets, times, tick=0, t_now=0.0, q_from=np.zeros(7))
    t2, times2 = chunk(0.7)
    s.set_chunk(t2, times2, tick=7, t_now=0.75, q_from=s.joint_target(0.75))   # before target 8 (0.8 s)
    assert s.held_s == 0.0
    s.set_chunk(*chunk(1.4), tick=14, t_now=1.62, q_from=s.joint_target(1.62))  # chunk 2 ended at 1.5 s
    assert abs(s.held_s - 0.12) < 1e-9


def simulate(exec_steps, delay, v=0.3, duration=6.0, rate=120.0):
    """Pure ChunkSchedule with a policy that knows the constant-velocity path q(t) = v t and perfect tracking:
    inference from tick k (every exec_steps ticks) returns targets on the path for t_k + 0.1 i, arriving
    `delay` s later, taking over from the current command. Returns (command speeds after the first chunk,
    held_s of every chunk)."""
    s = ChunkSchedule(exec_steps=exec_steps)
    direction = np.linspace(0.5, 1.0, 7) / np.linalg.norm(np.linspace(0.5, 1.0, 7))
    path = lambda t: v * t * direction
    arrivals = []
    for tick in range(0, int(duration * 10)):
        if s.chunk_tick is None or tick - s.chunk_tick >= exec_steps:
            t_obs = tick / 10
            times = t_obs + np.arange(1, 9) / 10
            targets = np.zeros((8, 8))
            targets[:, :7] = np.array([path(t) for t in times])
            arrivals.append((t_obs + delay, targets, times, tick))
            s.chunk_tick = tick      # due() counts from the tick the inference started from
    s.chunk_tick = None
    held, cmds, ts = [], [], np.arange(0, duration, 1 / rate)
    pending = list(arrivals)
    for t in ts:
        while pending and pending[0][0] <= t:
            t_arr, targets, times, tick = pending.pop(0)
            q_from = s.joint_target(t_arr)
            s.set_chunk(targets, times, tick, t_arr, path(t_arr) if q_from is None else q_from)
            held.append(s.held_s)
        cmds.append(s.joint_target(t))
    first = arrivals[0][0]
    keep = [i for i, t in enumerate(ts) if t > first + 0.05 and cmds[i] is not None and t < arrivals[-1][2][-1] - 0.05]
    q = np.array([cmds[i] for i in keep])
    speed = np.linalg.norm(np.diff(q, axis=0), axis=1) * rate
    return speed / v, held


def test_streaming_all_targets_removes_the_chunk_start_freeze_at_172_ms():
    speed6, held6 = simulate(exec_steps=6, delay=0.172)
    assert speed6.min() > 0.95 and speed6.max() < 1.05       # no stop, no lurch
    assert max(held6) == 0.0
    speed7, held7 = simulate(exec_steps=7, delay=0.172)
    assert max(held7) > 0.05                                  # 0.7 + 0.172 > 0.8: the command waits ~72 ms
    assert speed7.min() < 0.05


def test_sag_hold_chunks_stop_the_ratchet():
    """Toy plant: the arm settles s below whatever is commanded. A chunk-relative policy holding still commands
    the measured q (+ the offset). Without compensation the arm drops s per chunk; with it, ~0."""
    s_sag = np.full(7, 0.004)
    for max_rad, expect in ((0.0, -20 * s_sag), (0.015, np.zeros(7))):
        sag = SagCompensation(max_rad, mode="last_hold")
        q_meas = np.zeros(7)
        q_cmd = q_meas + s_sag                                  # already commanded and settled
        for _ in range(20):
            targets = np.tile(q_meas, (8, 1))                   # "hold still", anchored on the measured q
            offsets, offset, hold = sag.chunk(8, q_cmd, q_meas, speed=0.003, planned_motion_m=0.0008)
            assert hold
            q_cmd = targets[-1] + offsets[-1]
            q_meas = q_cmd - s_sag
        assert np.allclose(q_meas, expect, atol=1e-9)


def test_sag_moving_chunks_by_mode():
    q_meas = np.zeros(7)
    for mode, before_hold, after_hold in (("last_hold", 0.0, 0.004), ("none", 0.0, 0.0)):
        sag = SagCompensation(0.015, mode=mode)
        _, off, hold = sag.chunk(8, q_meas + 0.01, q_meas, speed=0.3, planned_motion_m=0.05)
        assert not hold and np.allclose(off, before_hold)      # moving before any hold
        _, off, hold = sag.chunk(8, q_meas + 0.004, q_meas, speed=0.001, planned_motion_m=0.001)
        assert hold and np.allclose(off, 0.004)
        offsets, off, hold = sag.chunk(8, q_meas + 0.02, q_meas, speed=0.3, planned_motion_m=0.05)
        assert not hold and np.allclose(off, after_hold)
        # the ramp: from the hold's 0.004 at the start to this chunk's offset at its last target
        assert np.allclose(offsets[0], 0.004 + (after_hold - 0.004) / 8) and np.allclose(offsets[-1], after_hold)
    # under last_hold, a move then a new hold ramps from the old hold's value to the fresh one
    sag = SagCompensation(0.015, mode="last_hold")
    sag.chunk(8, q_meas + 0.004, q_meas, speed=0.001, planned_motion_m=0.001)
    sag.chunk(8, q_meas + 0.02, q_meas, speed=0.3, planned_motion_m=0.05)
    offsets, off, _ = sag.chunk(8, q_meas + 0.006, q_meas, speed=0.001, planned_motion_m=0.001)
    assert np.allclose(offsets[0], 0.004 + 0.002 / 8) and np.allclose(off, 0.006)
    # clipped, never a hold before anything was commanded, off at 0
    _, off, _ = SagCompensation(0.015).chunk(8, q_meas + 0.05, q_meas, 0.001, 0.001)
    assert np.allclose(off, 0.015)
    assert not SagCompensation(0.015).chunk(8, None, q_meas, 0.001, 0.001)[2]
    assert np.allclose(SagCompensation(0.0).chunk(8, q_meas + 0.01, q_meas, 0.001, 0.001)[1], 0.0)
    with pytest.raises(ValueError):
        SagCompensation(0.015, mode="constant")


def test_width_target_is_the_latest_reached_target():
    s = ChunkSchedule()
    assert s.joint_target(0.0) is None and s.width_target(0.0) is None
    targets, times = chunk(0.0)
    targets[:, 7] = np.linspace(0.08, 0.02, 8)
    s.set_chunk(targets, times, tick=0, t_now=0.0, q_from=np.zeros(7))
    assert s.width_target(0.05) is None
    assert s.width_target(times[3]) == targets[3, 7]
    assert s.width_target(times[3] + 0.05) == targets[3, 7]


def test_hysteresis():
    h = GripperHysteresis(close_below=0.045, open_above=0.065)
    assert h.update(None) is False
    assert h.update(0.05) is False       # between the thresholds: stays open
    assert h.update(0.044) is True
    assert h.update(0.06) is True        # between: stays closed
    assert h.update(0.066) is False
    widths = [0.084, 0.07, 0.05, 0.044, 0.046, 0.02, 0.019, 0.05, 0.064, 0.07, 0.084]
    assert hysteresis_events(widths) == [(3, "close"), (9, "open")]
    with pytest.raises(ValueError):
        GripperHysteresis(close_below=0.07, open_above=0.065)


@pytest.mark.parametrize("path, name", [
    ("/data/cup/checkpoints/dp_overfit10_030000/pretrained_model", "dp_overfit10_030000"),
    ("/data/cup/checkpoints/diffusion_visual_capacity", "diffusion_visual_capacity"),
    ("/data/cup/checkpoints/diffusion_visual/400000/pretrained_model", "diffusion_visual@400000"),
    ("/supercloud/runs/dp_crop_a1/checkpoints/400000/pretrained_model", "dp_crop_a1@400000"),
])
def test_checkpoint_name(path, name):
    from policy_runner import checkpoint_name   # pure path logic: no checkpoint needed

    assert checkpoint_name(path) == name


@pytest.fixture(scope="module")
def runner():
    if not os.path.isdir(CKPT):
        pytest.skip(f"no checkpoint at {CKPT}")
    from policy_runner import PolicyRunner

    return PolicyRunner(CKPT, seed=0)


def images(runner, value=0):
    return {cam: np.full((*runner.image_hw[cam], 3), value, dtype=np.uint8) for cam in runner.cameras}


def test_runner_duplicates_the_first_frame_and_times_the_targets(runner):
    runner.reset()
    state = np.concatenate([np.linspace(-1, 1, 7), [0.084]])
    runner.push(images(runner, 7), state, 2.0)
    frames = list(runner._history)
    assert len(frames) == runner.n_obs_steps and all(f is frames[0] for f in frames)
    targets, times, latency = runner.infer()
    assert targets.shape == (8, 8) and latency > 0
    assert np.allclose(times, 2.0 + np.arange(1, 9) / 10)
    runner.push(images(runner, 9), state, 2.1)
    assert runner._history[0][2] == 2.0 and runner._history[1][2] == 2.1


def test_runner_rejects_bad_inputs(runner):
    runner.reset()
    state = np.zeros(8)
    runner.push(images(runner), state, 0.0)
    with pytest.raises(ValueError, match="ms apart"):
        runner.push(images(runner), state, 0.2)
    with pytest.raises(ValueError, match="expected images"):
        runner.push({runner.cameras[0]: images(runner)[runner.cameras[0]]}, state, 0.1)
    bad = images(runner)
    bad[runner.cameras[0]] = bad[runner.cameras[0]].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        runner.push(bad, state, 0.1)
    with pytest.raises(ValueError, match="state"):
        runner.push(images(runner), np.zeros(7), 0.1)
    runner.push(images(runner), state, 0.1 + 0.02)  # within tolerance


@pytest.fixture(scope="module")
def process_runner():
    if not os.path.isdir(CKPT):
        pytest.skip(f"no checkpoint at {CKPT}")
    from policy_process import PolicyProcess

    p = PolicyProcess(CKPT, seed=0)
    yield p
    p.close()


def test_process_matches_the_in_process_runner(process_runner):
    """Same seed, same observations -> the same plans (the model runs identically on the other side of the pipe)."""
    from policy_runner import PolicyRunner

    local = PolicyRunner(CKPT, seed=0)
    rng = np.random.default_rng(1)
    process_runner.reset(); local.reset()
    outs = []
    for i in range(3):
        imgs = {cam: rng.integers(0, 256, (*local.image_hw[cam], 3), dtype=np.uint8) for cam in local.cameras}
        state = np.r_[rng.normal(0, 0.3, 7), 0.084]
        process_runner.push(imgs, state, 0.1 * i); local.push(imgs, state, 0.1 * i)
        outs.append((process_runner.infer(), local.infer()))
    for (tp, timp, _), (tl, timl, _) in outs:
        assert np.allclose(timp, timl)
        assert np.allclose(tp, tl, atol=1e-4)
    assert process_runner.name == local.name and process_runner.n_action_steps == local.n_action_steps


def test_process_validates_in_the_caller_and_uses_the_latest_push(process_runner):
    r = process_runner
    r.reset()
    imgs = {cam: np.zeros((*r.image_hw[cam], 3), dtype=np.uint8) for cam in r.cameras}
    r.push(imgs, np.zeros(8), 1.0)
    with pytest.raises(ValueError, match="ms apart"):
        r.push(imgs, np.zeros(8), 1.2)
    r.push(imgs, np.zeros(8), 1.1)
    _, times, _ = r.infer()
    assert np.allclose(times, 1.1 + np.arange(1, 9) / 10)      # the plan is from the observation pushed last
    r.push(imgs, np.zeros(8), 1.2)
    assert np.allclose(r.infer()[1], 1.2 + np.arange(1, 9) / 10)
