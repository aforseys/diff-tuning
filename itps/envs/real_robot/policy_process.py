"""PolicyRunner in its own process: the same push / infer / reset interface, with the model on the other side of a
pipe, so inference doesn't compete with the robot loop for the Python interpreter (the GIL).

Why (2026-11-22, SimpleTrajectoryRecorder EVAL_FIXES.md): in a background THREAD of the rollout loop,
ebm_visual took 64-68 ms per plan against 48 ms alone; it lost the GIL to the loop's observation reads and
image processing. Its plans then arrived ~145 ms after their observation, too late for exec_steps 7.

    runner = PolicyProcess(".../pretrained_model")      # starts the process, loads the model there, waits
    runner.reset(); runner.push(images, state, t); targets, times, latency_s = runner.infer()
    runner.close()                                     # also at exit

ORDER: pushes, resets and plan requests go to the child over ONE pipe, so they are applied in the order sent.
A reader thread in the child applies each push and reset at once, and answers a plan request by taking a
snapshot of the history right then -- so a plan always uses exactly the observations pushed before it was
asked for, even if more arrive while it runs. The child's main thread runs the model.

VALIDATION happens in the parent (policy_runner.check_observation, as PolicyRunner.push), so a bad observation
-- e.g. two ticks 131 ms apart -- raises ValueError at push() in the caller, as before.
"""
import atexit
import multiprocessing as mp
import queue
import threading
import time
import traceback

import numpy as np

STARTUP_TIMEOUT_S = 180.0      # loading a checkpoint onto the GPU
INFER_TIMEOUT_S = 10.0         # a plan takes < 0.1 s; this only catches a dead or hung child
SPEC_ATTRS = ("name", "policy_type", "fps", "n_obs_steps", "n_action_steps", "image_keys", "cameras", "image_hw",
              "state_dim")


def _child(conn_in, conn_out, pretrained_model_dir, device, seed):
    try:
        from itps.envs.real_robot.policy_runner import PolicyRunner

        runner = PolicyRunner(pretrained_model_dir, device=device, seed=seed)
        spec = {k: getattr(runner, k) for k in SPEC_ATTRS}
        spec["device"] = str(runner.device)
        conn_out.send(("ready", spec))
    except BaseException:
        conn_out.send(("error", traceback.format_exc()))
        return
    requests = queue.Queue()

    def reader():
        try:
            while True:
                msg = conn_in.recv()
                kind = msg[0]
                if kind == "push":
                    runner.push(*msg[1:])
                elif kind == "reset":
                    runner.reset()
                elif kind == "infer":
                    requests.put(runner.snapshot())
                elif kind == "stop":
                    break
        except EOFError:
            pass
        except BaseException:
            requests.put(("error", traceback.format_exc()))
        requests.put(None)

    threading.Thread(target=reader, name="policy-process-reader", daemon=True).start()
    while True:
        item = requests.get()
        if item is None:
            break
        if isinstance(item, tuple) and item and item[0] == "error":
            conn_out.send(item)
            break
        try:
            conn_out.send(("result", runner.infer(frames=item)))
        except BaseException:
            conn_out.send(("error", traceback.format_exc()))


class PolicyProcess:
    """PolicyRunner's interface, the model in a child process (spawned: CUDA can't be forked)."""

    def __init__(self, pretrained_model_dir, device=None, seed=None):
        ctx = mp.get_context("spawn")
        self._to_child, child_in = ctx.Pipe(duplex=False)[::-1]     # (send end, receive end) -> child reads child_in
        child_out_recv, self._from_child_send = ctx.Pipe(duplex=False)
        self._from_child = child_out_recv
        self._process = ctx.Process(target=_child, name="policy-inference", daemon=True,
                                    args=(child_in, self._from_child_send, str(pretrained_model_dir), device, seed))
        self._process.start()
        child_in.close()                     # the child's ends: only the child keeps them open
        self._from_child_send.close()
        self._send_lock = threading.Lock()   # push (main thread) and infer (worker thread) share the pipe
        self._last_t = None
        kind, payload = self._receive(STARTUP_TIMEOUT_S, "loading the model")
        if kind != "ready":
            self.close()
            raise RuntimeError(f"the inference process failed to start:\n{payload}")
        for k, v in payload.items():
            setattr(self, k, v)
        self.pid = self._process.pid
        atexit.register(self.close)

    def _send(self, msg):
        with self._send_lock:
            self._to_child.send(msg)

    def _receive(self, timeout, what):
        deadline = time.monotonic() + timeout
        while not self._from_child.poll(0.05):
            if not self._process.is_alive():
                raise RuntimeError(f"the inference process died while {what} (exit code {self._process.exitcode})")
            if time.monotonic() > deadline:
                raise TimeoutError(f"the inference process took over {timeout:.0f} s {what}")
        try:
            return self._from_child.recv()
        except EOFError:
            raise RuntimeError(f"the inference process exited while {what} (its traceback is printed above; a "
                               f"script that starts one needs `if __name__ == \"__main__\":`)") from None

    def reset(self):
        self._last_t = None
        self._send(("reset",))

    def push(self, policy_images, state, t):
        from itps.envs.real_robot.policy_runner import check_observation

        state = check_observation(self, policy_images, state, t, self._last_t)   # raises here, in the caller
        self._last_t = float(t)
        self._send(("push", {cam: np.ascontiguousarray(img) for cam, img in policy_images.items()}, state, float(t)))

    def infer(self):
        """-> (targets, times, latency_s) as PolicyRunner.infer; latency_s is the model's own time in the child."""
        self._send(("infer",))
        kind, payload = self._receive(INFER_TIMEOUT_S, "computing a plan")
        if kind != "result":
            raise RuntimeError(f"inference failed in the inference process:\n{payload}")
        return payload

    def close(self):
        process = getattr(self, "_process", None)
        if process is None:
            return
        self._process = None
        try:
            with self._send_lock:
                self._to_child.send(("stop",))
        except (OSError, ValueError, BrokenPipeError):
            pass
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


INFERENCE_MODES = ("process", "thread")


def start_policy(pretrained_model_dir, device=None, seed=None, mode="process"):
    """The rollout's policy: a PolicyProcess ("process", the default) or an in-process PolicyRunner ("thread":
    run_episode calls infer() on a background thread, as before 2026-11-22)."""
    if mode == "process":
        return PolicyProcess(pretrained_model_dir, device=device, seed=seed)
    if mode == "thread":
        from itps.envs.real_robot.policy_runner import PolicyRunner

        return PolicyRunner(pretrained_model_dir, device=device, seed=seed)
    raise ValueError(f"inference mode must be one of {INFERENCE_MODES}, not {mode!r}")
