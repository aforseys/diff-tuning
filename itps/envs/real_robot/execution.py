"""When and what to command between policy chunks: the ONE definition, shared by the robot loop
(SimpleTrajectoryRecorder scripts/cup_demos/run_policy.py and dry_run_policy.py) and replay_check.py.

Pure numpy, no robot code, so it is tested offline.

Joints -- ChunkSchedule. The policy returns targets for t_obs + 0.1 i (i = 1..8), each the expected MEASURED
state at that time. The arm is streamed a joint target at the control rate (120 Hz, as the demos were flown),
interpolated linearly so the command passes through target_i exactly at its time:

    knots: (t_now, q_from), (t_1, target_1), ..., (t_8, target_8)     every target, past ones dropped

q_from is what the current chunk commands at the arrival time t_now (q_now for the first chunk), so the command
never jumps. EVERY target of the chunk is streamed; exec_steps only sets when the next inference starts (from
the observation exec_steps ticks after this chunk's). Inference runs in the background while the current chunk
streams on, and the new chunk takes over when it arrives. The chunk lasts to t_obs + 0.8 s, so the arm never
stops at a chunk boundary as long as exec_steps / fps + the arrival delay < n_action_steps / fps; otherwise the
command holds the last target until the next chunk arrives, and set_chunk records how long in `held_s`.
With ~172 ms from observation to arrival (the capacity model) that needs exec_steps <= 6; ~70 ms works with 7
(EVAL_FIXES.md F6). A chunk that arrives late just loses the targets whose time has passed.

Gripper -- GripperHysteresis on each streamed target's width, at its time: close below CLOSE_BELOW_M, open above
OPEN_ABOVE_M (measured open ~0.084 m, holding the cup ~0.019 m). The robot side commands fully open / closed.

Sag -- SagCompensation (EVAL_FIXES.md F8 / F10). The impedance tracker settles a little short of whatever it is
commanded, and the targets are measured states anchored on the measured q, so a "hold still" chunk would
ratchet the arm down. On a HOLD chunk (arm still and the plan barely moves) the offset is measured fresh as
clip(q_cmd - q_meas); any other chunk gets the last hold's offset ("last_hold") or none ("none"). Each chunk
ramps from the previous chunk's offset to its own across its targets.
"""
import numpy as np

N_JOINTS = 7
CLOSE_BELOW_M = 0.045
OPEN_ABOVE_M = 0.065
# Re-infer every this many 10 Hz ticks: the robot scripts' default (run_policy.py / dry_run_policy.py
# --exec_steps), and the default of the offline scorers (replay_check.py, eval_offline.py) so they replay
# the same cadence. The capacity model passes 6 (EVAL_FIXES.md F6).
DEFAULT_EXEC_STEPS = 7
SAG_MODES = ("last_hold", "none")


class ChunkSchedule:
    def __init__(self, exec_steps=DEFAULT_EXEC_STEPS):
        if exec_steps < 1:
            raise ValueError("exec_steps must be >= 1")
        self.exec_steps = int(exec_steps)
        self.chunk_tick = None
        self.held_s = 0.0          # how long the command sat on the previous chunk's last target (last set_chunk)
        self._knot_t = self._knot_q = None
        self._width_t = self._width = None

    def due(self, tick):
        """Whether observation tick `tick` should be fed to a new inference."""
        return self.chunk_tick is None or tick - self.chunk_tick >= self.exec_steps

    def set_chunk(self, targets, times, tick, t_now, q_from):
        """Start executing a chunk inferred from observation tick `tick`: all of its targets.

        targets (n, 8) absolute, times (n,) seconds; t_now is the time the chunk takes over, q_from (7,) the joint
        target commanded at that moment (joint_target(t_now) of the chunk it replaces). Returns the number of
        targets dropped because their time had passed; sets held_s.
        """
        targets = np.asarray(targets, dtype=float)
        times = np.asarray(times, dtype=float)
        self.held_s = 0.0 if self._knot_t is None else max(0.0, float(t_now) - float(self._knot_t[-1]))
        keep = times > t_now
        self._knot_t = np.concatenate([[float(t_now)], times[keep]])
        self._knot_q = np.vstack([np.asarray(q_from, dtype=float).reshape(1, N_JOINTS), targets[keep, :N_JOINTS]])
        self._width_t, self._width = times[keep], targets[keep, N_JOINTS]
        self.chunk_tick = int(tick)
        return int((~keep).sum())

    @property
    def pending_targets(self):
        """(k, 7) joint targets this chunk will still pass through -- what the arrival-time checks look at."""
        return None if self._knot_q is None else self._knot_q[1:]

    def joint_target(self, t):
        """(7,) joint command at time t, or None before the first chunk. Holds the last knot after it."""
        if self._knot_t is None:
            return None
        return np.array([np.interp(t, self._knot_t, self._knot_q[:, j]) for j in range(N_JOINTS)])

    def width_target(self, t):
        """Width of the latest executed target whose time has come, or None if none has yet."""
        if self._width_t is None:
            return None
        reached = np.nonzero(self._width_t <= t)[0]
        return None if len(reached) == 0 else float(self._width[reached[-1]])


class GripperHysteresis:
    def __init__(self, close_below=CLOSE_BELOW_M, open_above=OPEN_ABOVE_M, closed=False):
        if not close_below < open_above:
            raise ValueError("close_below must be below open_above")
        self.close_below, self.open_above = float(close_below), float(open_above)
        self.closed = bool(closed)

    def update(self, width):
        """Feed one executed target width (None = nothing new). Returns the state: True = closed."""
        if width is not None:
            if not self.closed and width < self.close_below:
                self.closed = True
            elif self.closed and width > self.open_above:
                self.closed = False
        return self.closed


def hysteresis_events(widths, close_below=CLOSE_BELOW_M, open_above=OPEN_ABOVE_M):
    """[(index, "close" | "open")] where a width sequence (starting open) switches state."""
    h, events = GripperHysteresis(close_below, open_above), []
    for i, w in enumerate(widths):
        before = h.closed
        if h.update(float(w)) != before:
            events.append((i, "close" if h.closed else "open"))
    return events


class SagCompensation:
    """The per-chunk sag offset (EVAL_FIXES.md F8 / F10). Pure: the caller measures speed and planned motion.

    max_rad: per-joint clip on the offset (rollout.sag_compensation_max_rad); 0 turns it off.
    mode: what a chunk that is not a hold gets -- "last_hold": the offset measured on the most recent hold
        chunk (0 before the first), "none": 0.
    A HOLD chunk: the arm is still at the chunk's observation (fastest joint below hold_max_speed rad/s) and the
    plan barely moves (largest gripper distance of any target from the anchor below hold_max_motion_m). Its
    offset is measured fresh, clip(q_cmd - q_meas, +-max_rad); back-to-back holds stay put because q_cmd
    already includes the previous offset.
    """

    def __init__(self, max_rad, mode="last_hold", hold_max_speed=0.02, hold_max_motion_m=0.003):
        if mode not in SAG_MODES:
            raise ValueError(f"sag mode must be one of {SAG_MODES}, not {mode!r}")
        self.max_rad, self.mode = float(max_rad), mode
        self.hold_max_speed, self.hold_max_motion_m = float(hold_max_speed), float(hold_max_motion_m)
        self.last_hold = np.zeros(N_JOINTS)   # the offset measured on the most recent hold chunk
        self.prev = np.zeros(N_JOINTS)        # the offset the previous chunk ended on

    def is_hold(self, speed, planned_motion_m):
        return float(speed) < self.hold_max_speed and float(planned_motion_m) < self.hold_max_motion_m

    def chunk(self, n, q_cmd, q_meas, speed, planned_motion_m):
        """For a chunk of n targets: ((n, 7) offsets to add to its joint targets -- a ramp from the previous
        chunk's offset to this one's -- this chunk's offset (7,), hold). q_cmd None (nothing commanded yet)
        is never a hold."""
        hold = q_cmd is not None and self.is_hold(speed, planned_motion_m)
        if self.max_rad <= 0:
            offset = np.zeros(N_JOINTS)
        elif hold:
            offset = np.clip(np.asarray(q_cmd, dtype=float) - np.asarray(q_meas, dtype=float),
                             -self.max_rad, self.max_rad)
            self.last_hold = offset
        else:
            offset = self.last_hold.copy() if self.mode == "last_hold" else np.zeros(N_JOINTS)
        ramp = (np.arange(1, n + 1) / n)[:, None]
        offsets = self.prev + (offset - self.prev) * ramp
        self.prev = offset
        return offsets, offset, hold
