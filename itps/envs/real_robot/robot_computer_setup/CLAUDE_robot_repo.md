# Standing instructions: robot repo, real-robot work

(Project-level only: this file is `<robot repo>/CLAUDE.md`. Sessions for real-robot work start in the robot repo, with the diff-tuning clone added via `/add-dir`.)

## The robot: Claude never moves it
- **Claude never runs anything that can move the arm or gripper.** That includes:
  - rollout scripts, demo playback, collection, teleop and recording scripts;
  - homing and the cup-placement pointer mode;
  - any one-off `python -c` or test that calls motion or gripper commands.

  The user runs all of these. Claude writes the code and hands over the exact command to run.
- **What Claude may run:** offline checks, camera captures, robot-state reads, `place_cup.py` and `dry_run_policy.py`.
  - These must stay structurally unable to move: no tracker, `robot.move`, `move_to_start` or `move_width` reachable from them.
  - If a change would break that, say so instead of making it.
- **Never loosen a safety check, threshold, joint margin, table height or episode cap** to get a rollout through. Propose the change with the evidence; the user decides.
- **After a safety trip or a failed rollout:** analyze the logs first. Don't suggest re-running until the cause is understood.
- **Before editing control-loop, safety, gripper or tracker code,** explain the change.

## The repos
- **diff-tuning** (`github.com/aforseys/diff-tuning`, a LeRobot fork): visual Diffusion Policy (`diffusion`) and an energy-based variant (`ebm_diffusion`), trained on MIT Supercloud.
  - The real-robot inference code is in `itps/envs/real_robot/`.
  - `image_pipeline.py` is the ONLY image path.
  - `chunk_relative.py` handles action anchoring.
  - **The current task and the full policy interface are in `itps/envs/real_robot/ROLLOUT_PLAN.md`.** Read it before any real-robot work.
  - Its "settled decisions" are not re-opened without the user.
- **The robot repo** (SimpleTrajectoryRecorder-based: zero_franky, the Robotiq gripper, ZED cameras) holds:
  - the hardware loop and demo collection (`scripts/cup_demos/`);
  - the tracker, playback and key helpers (`simple_recorder/playback.py`);
  - the kinematics (`simple_recorder/kinematics.py`);
  - the camera, gripper and joint-limit config (`conf/config.yaml`).

  Reuse these; don't reimplement camera, gripper or tracker code.
- **Training never runs here.** Supercloud jobs are submitted with `LLsub <script> [NODES,NPPN,NTPP] -g volta:2`. The `-g volta:2` is required, or the job gets no GPUs.

## Working with the user
- **Git:** never `git add`, commit or push without the user's explicit OK for that action.
- **Fixing a problem:** before editing code, explain the mechanism and check that it accounts for ALL the symptoms, reproducing them where possible.
- **While a design is still being discussed:** no edits or tool runs until the user says go.
- **A refactor never changes behavior or names.** If a change would, it isn't a refactor; say so.
- **Settling claims:**
  - shape, library and numeric claims: by running a small piece of code, not by reasoning;
  - simple algebra: a 2-line derivation.
- **The user's runs, plots and observations are ground truth.** Don't re-derive to argue with them.
- **Answer the question asked, cheaply:** no deep history or provenance hunts for a direct question.
- **Files outside the repo:** ask the user for paths rather than searching home directories.
- **Reuse existing helpers.** Keep data processing simple: compute and report directly, with no fixtures.
- **Keep replies short while debugging,** and rank findings by impact.
