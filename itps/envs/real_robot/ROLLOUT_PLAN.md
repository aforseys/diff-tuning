# Handoff plan: real-robot rollouts of the overfit-10 policies (for the robot-computer session)

Written for: the Claude agent on the robot command computer, which has no context from the training sessions.

**Handoff:** this file is the handoff from the training-side sessions. The robot-computer session writes all of the code below.

**Who runs what:** Claude never runs anything that can move the arm or gripper. The user runs every motion script, and Claude hands over the exact command. Claude may run the offline checks (A–B), the live image check (C), `dry_run_policy.py` and `place_cup.py`. None of these can move the robot, by construction (see D).

## Context
- **What was trained:** two visual policies (repo `github.com/aforseys/diff-tuning`, a LeRobot fork), both overfit for 30k steps on 10 demos of a single setup: cup start `s1_disc5` → shelf `e1`.
  - DP: `diffusion`.
  - EBM: `ebm_diffusion`, an energy-gradient noise prediction.
  - Both are on Supercloud.
- **Their purpose:** to develop and test the inference and rollout path. They should reproduce that one demo motion.
- **Nothing exists yet for real-time control.** The repo has the training side, plus the two pieces shared with rollout: `itps/envs/real_robot/image_pipeline.py` and `itps/common/datasets/chunk_relative.py`.
- **Later:** the 5 full-data base policies (297 demos, 3 shelves, 400k steps) are still training. Their checkpoint format is identical, so this code serves them too.
- **Settled decisions:** do not re-open these without the user.
  - DDIM with 10 steps.
  - Chunk-wise joint anchor.
  - Measured gripper width, thresholded on the robot side.
  - No JPEG round trip by default.

## 1. What to move onto the robot computer
| Item | From | Notes |
|---|---|---|
| Repo at ≥ `c8553b6` | `git clone git@github.com:aforseys/diff-tuning.git` | Make a separate Python env for it, e.g. `pip install -e .` with a CUDA torch build if there's a GPU. |
| DP and EBM EMA weights | Supercloud run dirs `itps/data/real_robot/train/2026.11.19/<time>_real_robot_{diffusion,ebm_diffusion}_visual_overfit10/checkpoints/030000/pretrained_model/` | `model.safetensors`, `config.json`, `config.yaml`, about 0.35 GB each. **Not** `training_state.pth` (optimizer plus online weights, only for resuming). |
| `cup_overfit10.hdf5` (168 MB) | Supercloud `itps/data/real_robot/` | The exact 10 Hz training inputs for these 10 demos, used for the replay check. |
| Already on this computer | — | Raw demos, the robot repo (`scripts/cup_demos/collect_demos.py`: camera config and serials, home pose, cup and shelf positions), zero_franky, the ZED SDK. |

```bash
SC=aforsey@txe1-login.mit.edu   # Supercloud login node; compute nodes are not reachable
R=/home/gridsan/aforsey/diff-tuning/itps/data/real_robot
for RUN in $(ssh $SC "cd $R/train && ls -d */*_overfit10"); do
  mkdir -p ~/policies/$(basename $RUN)
  rsync -av $SC:$R/train/$RUN/checkpoints/030000/pretrained_model ~/policies/$(basename $RUN)/
done
rsync -av $SC:$R/cup_overfit10.hdf5 ~/policies/
```
If `030000` is missing, use the step that `checkpoints/last` points to.

## 2. Interface contract (must match training exactly)
- **Loading:**
  1. `hydra_cfg = init_hydra_config(<pm>/config.yaml)` (`itps/common/utils/utils.py`).
  2. Set `hydra_cfg.device` to `cpu` if there's no GPU.
  3. `policy = make_policy(hydra_cfg, pretrained_policy_name_or_path=<pm>)` (`itps/common/policies/factory.py`). The normalization stats are inside the checkpoint.
  4. **`policy.eval()` is mandatory.** Training mode applies a random crop.
- **Cameras:** `observation.image.wrist` (ZED-M) and `observation.image.agent` (ZED 2i).
  - Use the same names, serials, HD720 (1280×720) resolution and settings as `collect_demos.py`.
  - `agent_right` is not used.
- **Image path (the only one):** `camera.read()` returns a BGR frame. Then:
  1. `image_pipeline.live_frame(bgr)`: true RGB 720×1280.
  2. `to_policy_image`: 4:3 center crop, INTER_AREA resize to 120×160 uint8.
  3. `to_model_input`: float32 (3,120,160) in [0,1].

  The policy does its own 108×144 center crop internally.
  - No other resizing, cropping or decoding anywhere.
  - Never decode raw demo JPEGs with PIL: the recorder swaps R and B. `decode_recorded` and the hdf5 are correct.
- **State:** `observation.state` = [q (7, measured: `robot.get_last_teleop_state()["q"]`), gripper width in m (measured: `gripper.position_width(refresh_status=True, max_width_m=0.085)`)]. These are the same calls the recorder's `log_step` makes.
- **History:** `n_obs_steps=2`, frames t−1 and t, **exactly 100 ms apart** (the 10 Hz grid). At episode start, duplicate the first frame, as training did.
- **Inference:**
  - Each key is shaped (1, 2, …) and goes on the device.
  - Call `policy.run_inference(obs, methods=("ddim",))`. The EBM's default methods include IRED, which needs `opt_params`, so always pass `("ddim",)`.
  - It returns `[a]` with `a` (1, 8, 8): **chunk-relative** labels for the frames t+1…t+8.
  - There is no `select_action` or action queue in this repo.
- **Absolute targets:** `targets = from_chunk_relative(a, obs_state[:, -1])`.
  - Joints are `q_obs[t] + label_i`. The width stays absolute.
  - The anchor is the q **inside the observation that was fed in**, not q at execution time.
  - Never use a cumsum.
  - Target i is the expected *measured* state at `t_obs + 0.1·i` s.

## 3. Code to write
**A. `itps/envs/real_robot/policy_runner.py`** (independent of the robot; lives in the repo so it can be tested offline)
- `PolicyRunner(pretrained_model_dir, device)` with:
  - `reset()`;
  - `push(policy_images: {cam: uint8 120×160×3 RGB}, state (8,), t)`, which keeps a 2-deep buffer and duplicates the first frame;
  - `infer()`, which returns `(targets (8,8) absolute, times t_obs+0.1·i, latency_s)`.
- It takes policy images, which are what the hdf5 stores. That way the replay check and the live path share everything after `to_policy_image`.

**B. `itps/envs/real_robot/replay_check.py`** (offline, run before any robot motion)
- For each demo in `cup_overfit10.hdf5` and each anchor t:
  - feed frames (t−1, t) from the file;
  - compare the predicted targets with the recorded `q[t+1..t+8]` and `w[t+1..t+8]`.
- **Report:**
  - per-joint RMSE for each chunk index i, next to the spread across the 10 demos at the same phase;
  - gripper close and open timing error, in frames;
  - mean and p95 inference latency, for DP and for the EBM.
- **Plots:** predicted versus recorded joint traces for one demo.
- **Pass:** errors comparable to the demo-to-demo spread, and gripper timing within about 1–2 frames.

**C. Live image check:** with the robot at home, capture both cameras through the live path. Save each next to the same camera's frame 0 from the hdf5. Check:
- the cup is red;
- the right camera is under the right key;
- the framing and brightness match.

Also print the mean absolute pixel difference.

**D. `scripts/cup_demos/run_policy.py` in the robot repo** (the control loop; reuse the recorder's code)
- **Setup, as in `collect_demos.py`:**
  - the cameras, the zero_franky robot and the Robotiq gripper;
  - `move_to_start` to `home.joints` with `JointMotion` at the home dynamics factors;
  - the gripper open.
- **10 Hz observation clock,** independent of inference:
  1. On each tick, read both cameras and q/width in `log_step` order.
  2. Pass the frames through `live_frame` and `to_policy_image`.
  3. `runner.push(...)`.
  4. Log everything.
- **Execution (DP-style, one chunk at a time):**
  - At obs tick t, call `infer()`.
  - Drop targets whose time is already past (the latency). Stream the rest through the joint impedance tracker: `playback.start_tracker(robot, cfg)` with replay stiffness and `period 0.001`, then `session.set_target(q_i)`. Hold each target until its time, as `play_trajectory.py` does; linear interpolation is an option.
  - Re-infer at the tick after target 8, so `exec_steps=8`, as trained. Add an `--exec_steps k` flag for replanning sooner.
- **Gripper:**
  - Threshold each executed target's width with hysteresis. Close below about 0.045 m and open above about 0.065 m (open is about 0.084, holding the cup about 0.019).
  - Command it the way demos were collected: `move_width(0.085 or 0.0, speed=255, force=255, wait=False)`, as in `gripper.toggle_gripper`.
- **Safety**, in a `safety.py` next to `run_policy.py`. Every check that trips is logged with its values. A trip holds the current q (`set_target(q_now)`) and ends the episode.
  - **Joint limits:** these already exist in the recorder config (`joint_limits.lower/upper`, `conf/config.yaml`), and `start_tracker` passes them to the tracker as hard limits. In addition, before sending each target, reject it if it comes within a soft margin of those limits (about 0.05 rad).
  - **Table plane, no IK needed.** The policy outputs joint targets, so checking them only takes forward kinematics, which is cheap: `simple_recorder.kinematics.fk(q_target, F_T_EE)`.
    - **Pre-send check:** reject any target whose predicted EE or fingertip height is below `z_stop`.
    - **Watchdog thread, at about 100 Hz,** independent of the policy loop. It reads the measured `O_T_EE` from `get_last_teleop_state()` and trips if z < `z_stop`. This also catches motion that tracks badly or overshoots, which the pre-send check can't see.
    - **Setting `z_stop`:**
      1. Teleop the fingertips onto the table at about 3 points and record `O_T_EE` z, giving `z_table`.
      2. Find which point `O_T_EE` refers to (see `F_T_EE`). If it isn't the fingertips, add the offset to them.
      3. Choose `z_stop` above `z_table` but below the lowest EE height in the `s1_disc5→e1` demos (the raw demos record `ee_positions`), minus a little. The grasp brings the fingers close to the table, so too high a value will trip on correct grasps.
      4. Save both values in the robot repo's config.
    - The shelves are not covered by this plane. Add boxes later if needed.
  - **Target jump:** reject any target more than 0.15 rad from the current q. The data's largest single 100 ms step is 0.012–0.040 rad.
  - **Episode cap:** 40 s (demos are about 26 s).
  - **Keyboard stop:** via `playback.RawTerminal`/`read_key`.
  - Someone on the e-stop.
  - **Dry run as its own entry point, `scripts/cup_demos/dry_run_policy.py`, not a flag:**
    - It runs the same observation and inference loop, with the same safety checks. It prints and plots the targets and reports which checks would have tripped.
    - It never constructs the tracker and never sends a gripper or motion command. No `start_joint_impedance_tracker`, `robot.move`, `move_to_start` or `move_width` is reachable from it.
    - A separate script makes "cannot move" structural, so Claude can safely run the dry run while every motion script stays the user's to run.
- **Per-rollout log:** policy images, q, width, every predicted chunk with timestamps, latency, gripper commands, outcome.

**E. `scripts/cup_demos/place_cup.py`** in the robot repo: a quick check, run before every rollout, for putting the cup exactly at the start.
- **References.** Every demo starts at the same home q, so frame 0 of each demo is a fixed view of where the cup should be.
  - Decode frame 0 from the 10 raw `s1_disc5→e1` demos at full 1280×720 with `image_pipeline.decode_recorded` (true RGB), for the agent and wrist cameras.
  - Take the per-pixel median as the reference image.
  - Find the red cup with an HSV threshold, and record its mask centroid and area in each demo.
  - Set the tolerance from the demos themselves: the largest centroid deviation among the 10.
- **Live check,** with the robot at home and both cameras through `live_frame`:
  - Show the live view blended 50/50 with the reference, and the outline of the reference cup mask.
  - Show the live cup centroid's offset in pixels from the reference, per camera, as an arrow.
  - Print PASS when both cameras are within tolerance. The run loop refuses to start without a PASS from just before.
  - Two viewpoints pin down x and y. The overlay also shows if the cup is rotated, if it has a handle.
- `place_cup.py` itself never moves anything: the user puts the robot at home first.
- **Optional pointer mode,** in its own script, `scripts/cup_demos/place_cup_pointer.py`, because it moves the arm. For setting up or when unsure:
  1. With the gripper open, move the arm to the demo's frame a few steps before `first_gripper_close_index` (`playback.py`), using `JointMotion`. That puts it just above the grasp.
  2. Place the cup between the fingers.
  3. Return home and run the live check.

## 4. Eval sequence (stop at the first failure)
1. Both checkpoints load. Run one inference on a dummy observation and note the latency on this computer, DP vs. EBM. If there's no GPU, check that latency fits well inside 0.8 s.
2. Replay check (B) passes for DP and the EBM.
3. Live image check (C) passes.
4. **Safety check without the policy.**
   - At home, the watchdog and pre-send checks report the margins (EE z − `z_stop`, distance to the joint limits).
   - Jog the EE down toward the table with teleop: the watchdog must trip above the table.
5. **Actuation test without the policy.** Stream one recorded `s1_disc5→e1` demo's 10 Hz q and width from the hdf5 through D's tracker and gripper-threshold path, with the cup in place. The grasp and place should succeed. This keeps actuation bugs separate from policy bugs.
   No safety check should trip during it.
6. `dry_run_policy.py` at home: the first targets should be near the current q, repeated chunks should be stable, and no check should trip.
7. Closed-loop rollouts, DP first, then the EBM. Each one starts with E's cup check passing at home.
   - The cup goes at `s1_disc5` and the target is shelf `e1`, the only trained setup.
   - Do 10 trials each.
   - Log: success, and the failure stage if not (reach / grasp / transport / place).
8. **Only if live performance is worse than replay:**
   - `live_frame(..., jpeg_roundtrip=True)`, which re-adds the recorder's quality-92 JPEG artifacts.
   - DDIM 10 vs. 100 steps, only if accuracy looks poor.

   Both were deferred by the user.
9. Optional: another cup start. This is expected to fail, because the model is overfit, and it confirms the policy uses the images.

## 5. Later, for the full-data base policies
- Same runner. Load `checkpoints/400000/pretrained_model` from `11.40.41_real_robot_diffusion_visual` and its sibling runs.
- **Offline diagnostics:**
  - Shelf-mode coverage: sample about 100 times at the `middle` phase and check the shelves split about 1/3 each.
  - Visual-grounding swap at home, t=0: the joints are identical across demos, so only the images can pick the shelf.
- `get_energy` does not support image stacking yet. Don't use it.

## Verification
- B's report and plots for both policies.
- C's side-by-side image.
- The logged safety and actuation tests (steps 4–5).
- 10 logged rollouts per policy with outcomes.

Each stage's logs are saved, so failures can be replayed offline through `PolicyRunner`.
