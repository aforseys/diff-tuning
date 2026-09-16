#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Evaluation for the robosuite bin-placing environment: rolls the policy out in MuJoCo and reports
placement success plus trajectory features."""
import numpy as np
import torch

from itps.common.utils.utils import seeded_context


def _robosuite_conditions(is_goal_cond, train_prisms, test_prisms, train_bins, test_bins):
    """Return list of (label, prism_list, bin_list_or_None) eval conditions."""
    conditions = []
    if is_goal_cond:
        combos = [
            ('seen_start_seen_goal', train_prisms, train_bins),
            ('new_start_seen_goal',  test_prisms,  train_bins),
            ('seen_start_new_goal',  train_prisms, test_bins),
            ('new_start_new_goal',   test_prisms,  test_bins),
        ]
        for label, prisms, bins in combos:
            if prisms is not None and bins is not None:
                conditions.append((label, prisms, bins))
    else:
        if train_prisms is not None:
            conditions.append(('train', train_prisms, None))
        if test_prisms is not None:
            conditions.append(('test', test_prisms, None))
    return conditions


def _sample_episode_indices(obs_data, prisms, bins, n_episodes, is_goal_cond, rng):
    """Sample n_episodes row indices from obs_data matching the given prism/bin lists."""
    prism_idx = obs_data['prism_idx']
    bin_idx   = obs_data['bin_idx']

    mask = np.isin(prism_idx, prisms)
    if is_goal_cond:
        mask &= np.isin(bin_idx, bins)
    else:
        mask &= (bin_idx == 0)   # deduplicate: each start config appears once per bin

    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        bin_desc = f"bin_idx==0" if bins is None else f"bins={bins}"
        raise ValueError(f"No observations found for prisms={prisms}, {bin_desc}")
    return rng.choice(candidates, size=n_episodes, replace=len(candidates) < n_episodes)


def _viz_sampled_trajectories(env, obs, policy, obs_batch, n_viz_samples, n_joints, chunk_size, start_idx, device):
    """Sample n_viz_samples trajectories, place sphere markers at EEF waypoints, render at high res."""
    import mujoco
    import matplotlib.pyplot as plt
    import colorsys

    # Batch obs n_viz_samples times → different noise initializations → different trajectories
    batched = {k: v.repeat(n_viz_samples, *([1] * (v.dim() - 1))) for k, v in obs_batch.items()}
    with torch.no_grad():
        _, full_trajs = policy.run_inference(batched, methods=['ddim'], return_full=True)
    trajs = full_trajs[0][:, start_idx:, :7].cpu().numpy()  # (N, T, 7) — full predicted future

    # FK: compute EEF path for each sample without stepping physics
    saved_qpos = env.sim.data.qpos.copy()
    start_joints = obs["robot0_joint_pos"].copy()
    all_eef = []
    for i in range(n_viz_samples):
        eef_path = []
        current = start_joints.copy()
        for t in range(trajs.shape[1]):
            current = current + trajs[i, t]
            env.sim.data.qpos[:n_joints] = current
            env.sim.forward()
            eef_path.append(env._eef_pos().copy())
        all_eef.append(np.array(eef_path))
    env.sim.data.qpos[:] = saved_qpos
    env.sim.forward()

    # Use the existing offscreen render context — inject sphere geoms before rendering
    VIZ_W, VIZ_H = 640, 480
    cam_id = env.sim.model.camera_name2id('agentview')
    ctx = env.sim._render_context_offscreen
    ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    ctx.cam.fixedcamid = cam_id

    mujoco.mjv_updateScene(
        env.sim.model._model, env.sim.data._data,
        ctx.vopt, ctx.pert, ctx.cam, mujoco.mjtCatBit.mjCAT_ALL, ctx.scn
    )

    # Inject sphere geoms — subsample to ~10 waypoints per trajectory
    step = max(1, trajs.shape[1] // 15)
    for i, eef_path in enumerate(all_eef):
        h = i / n_viz_samples
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
        rgba = np.array([r, g, b, 0.8], dtype=np.float32)
        for pos in eef_path[::step]:
            if ctx.scn.ngeom >= ctx.scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                ctx.scn.geoms[ctx.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.012, 0.012, 0.012]),
                pos.astype(np.float64),
                np.eye(3, dtype=np.float64).flatten(),
                rgba,
            )
            ctx.scn.ngeom += 1

    # Render directly — don't use ctx.render() which would call mjv_updateScene again and wipe our geoms
    ctx.update_offscreen_size(VIZ_W, VIZ_H)
    viewport = mujoco.MjrRect(0, 0, VIZ_W, VIZ_H)
    mujoco.mjr_render(viewport, ctx.scn, ctx.con)
    img = ctx.read_pixels(VIZ_W, VIZ_H)[::-1]  # flip bottom-up → top-down

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'{n_viz_samples} sampled EEF trajectories', fontsize=13)
    plt.tight_layout()
    plt.show(block=True)


def eval_robosuite(policy, cfg, seed=None, render=False, n_viz_samples=0):
    """
    Evaluate the robosuite policy against fixed observations from cfg.eval.obs_file.

    Config fields (under eval):
      obs_file:      path to .npz with start_pos, goal_pos, bin_idx, prism_idx, start_joints
      train_prisms:  list of prism indices seen during finetuning
      test_prisms:   list of prism indices NOT seen during finetuning
      train_bins:    list of bin indices seen during finetuning (GC only; null for non-GC)
      test_bins:     list of bin indices NOT seen during finetuning (GC only; null for non-GC)
      n_episodes:    episodes sampled per condition
    """
    if seed is None:
        return _eval_robosuite(policy, cfg, seed, render, n_viz_samples)
    # See eval_maze above: seeded_context restores the caller's RNG on exit, so an
    # in-training eval doesn't reseed training's noise stream.
    with seeded_context(seed):
        return _eval_robosuite(policy, cfg, seed, render, n_viz_samples)


def _eval_robosuite(policy, cfg, seed, render, n_viz_samples):
    rng = np.random.default_rng(seed)

    from collections import deque
    from itps.envs.robosuite.bin_placing import make_eval_env, OBJECT_MAP

    n_episodes       = cfg.eval.n_episodes
    n_bins           = 4
    n_obs_steps      = policy.config.n_obs_steps
    steps_per_action = cfg.eval.get('steps_per_action', 2)
    is_goal_cond     = 'episode_goal' in policy.config.input_shapes
    chunk_sizes      = list(cfg.eval.get('action_chunk_sizes', [policy.config.n_action_steps]))
    max_steps        = cfg.eval.get('max_episode_steps', 300)
    img_size         = cfg.eval.get('img_size', 84)
    obj              = OBJECT_MAP.get(cfg.eval.get('object', 'can'))
    device           = next(policy.parameters()).device

    obs_file     = cfg.eval.get('obs_file', None)
    train_prisms = list(cfg.eval.get('train_prisms')) if cfg.eval.get('train_prisms') else None
    test_prisms  = list(cfg.eval.get('test_prisms'))  if cfg.eval.get('test_prisms')  else None
    train_bins   = list(cfg.eval.get('train_bins'))   if cfg.eval.get('train_bins')   else None
    test_bins    = list(cfg.eval.get('test_bins'))    if cfg.eval.get('test_bins')    else None

    if obs_file is None:
        raise ValueError("cfg.eval.obs_file must be set for robosuite eval")

    obs_data = np.load(obs_file)
    n_joints = obs_data['start_joints'].shape[1]

    metrics      = list(cfg.eval.get('metrics', []))
    eval_methods = list(cfg.eval.get('methods', ['ddim']))
    opt_params   = list(cfg.eval.get('opt_params') or [])
    if 'ired' in eval_methods and not opt_params:
        raise ValueError("eval.methods includes 'ired' but eval.opt_params is empty; give one dict per IRED variant.")

    method_variants = []
    if 'ired' in eval_methods:
        for i, op in enumerate(opt_params):
            label = f"ired_{op['n_opt']}steps"
            if op.get('t_subset') is not None:
                label += f"_last{op['t_subset']}"
            if op.get('denoise'):
                label += '_denoise'
            method_variants.append((label, i))
    if 'ddim' in eval_methods:
        ddim_idx = len(opt_params) if 'ired' in eval_methods else 0
        method_variants.append(('ddim', ddim_idx))

    conditions = _robosuite_conditions(is_goal_cond, train_prisms, test_prisms, train_bins, test_bins)
    if not conditions:
        raise ValueError("No eval conditions found — set train_prisms/test_prisms (and train_bins/test_bins for GC)")

    state_dim = policy.config.input_shapes["observation.state"][0]
    past_action_visible = state_dim > 8

    def get_state(obs, gripper_cmd, prev_action=None):
        state = np.concatenate([obs["robot0_joint_pos"], [gripper_cmd]]).astype(np.float32)
        if past_action_visible:
            pa = prev_action if prev_action is not None else np.zeros(8, dtype=np.float32)
            state = np.concatenate([state, pa])
        return state

    def get_image(obs):
        img = obs["agentview_image"].astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)  # (3, H, W)

    def get_placed_bin(env):
        result = {'location': None, 'placement': None}
        for b in range(n_bins):
            if env.location_success(b):
                result['location'] = b
                if env.placement_success(b):
                    result['placement'] = b
                break
        return result

    all_metrics = {}
    env = make_eval_env(img_size=img_size, mujoco_object=obj, render=render)

    for method_label, method_traj_idx in method_variants:
      for chunk_size in chunk_sizes:
        rng = np.random.default_rng(seed)
        for cond_label, prisms, bins in conditions:
            indices = _sample_episode_indices(obs_data, prisms, bins, n_episodes, is_goal_cond, rng)
            location_bins, placement_bins, target_bins = [], [], []
            feat_scores = {m: [] for m in metrics}

            for ep_i, idx in enumerate(indices):
                target_bin  = int(obs_data['bin_idx'][idx])
                joint_start = obs_data['start_joints'][idx]

                obs = env.reset()
                env.sim.data.qpos[:n_joints] = joint_start
                env.sim.data.qvel[:n_joints] = 0.0
                env.sim.forward()
                # Re-place object at new EEF so the robot is grasping it
                # (_reset_internal placed it at the default rest-pose EEF, which moved when we teleported)
                eef_pos = env._eef_pos()
                env.sim.data.set_joint_qpos(
                    env.grasp_obj.joints[0],
                    np.concatenate([eef_pos, np.array([1, 0, 0, 0])])
                )
                env.sim.forward()
                # Hold arm at joint_start via full controller while closing gripper to stabilize grasp
                hold_act = np.concatenate([joint_start, [1.0]])
                for _ in range(50):
                    env.step(hold_act)
                obs, _, _, _ = env.step(hold_act)

                gripper_cmd = 1.0
                prev_action = np.zeros(8, dtype=np.float32)
                state_buf = deque([get_state(obs, gripper_cmd, prev_action)] * n_obs_steps, maxlen=n_obs_steps)
                image_buf = deque([get_image(obs)]                            * n_obs_steps, maxlen=n_obs_steps)

                if is_goal_cond:
                    goal = np.zeros(n_bins, dtype=np.float32)
                    goal[target_bin] = 1.0
                    goal_tensor = torch.tensor(goal).unsqueeze(0).to(device)

                eef_pos_buf  = []
                eef_quat_buf = []

                step = 0
                while step < max_steps:
                    obs_batch = {
                        'observation.state':
                            torch.tensor(np.stack(state_buf), dtype=torch.float32).unsqueeze(0).to(device),
                    }
                    if 'observation.image.agentview' in policy.config.input_shapes:
                        obs_batch['observation.image.agentview'] = \
                            torch.tensor(np.stack(image_buf), dtype=torch.float32).unsqueeze(0).to(device)
                    if is_goal_cond:
                        obs_batch['episode_goal'] = goal_tensor

                    if ep_i == 0 and step == 0:
                        print(f"=== EVAL DEBUG (ep={ep_i}, step={step}) ===")
                        print("obs_batch shapes:", {k: v.shape for k, v in obs_batch.items()})
                        print("state sample (first obs step):", obs_batch["observation.state"][0, 0])
                        if "observation.image.agentview" in obs_batch:
                            img = obs_batch["observation.image.agentview"][0, 0]
                            print(f"image range: [{img.min():.3f}, {img.max():.3f}]")
                        print("=================================")

                    if n_viz_samples > 0 and step == 0:
                        _viz_sampled_trajectories(env, obs, policy, obs_batch, n_viz_samples, n_joints, chunk_size, policy.config.n_obs_steps - 1, device)

                    with torch.no_grad():
                        _, full_trajs = policy.run_inference(obs_batch, methods=eval_methods, opt_params=opt_params, return_full=True)
                    start = policy.config.n_obs_steps - 1
                    chunk = full_trajs[method_traj_idx][0][start:start + chunk_size].cpu().numpy()

                    if ep_i == 0 and step == 0:
                        print(f"=== ACTION DEBUG (chunk_size={chunk_size}) ===")
                        print(f"full_traj shape: {full_trajs[0][0].shape}")
                        print(f"chunk[0] (first action): {chunk[0]}")
                        print(f"chunk mean: {chunk.mean(axis=0).round(4)}")
                        print(f"chunk std:  {chunk.std(axis=0).round(4)}")
                        print(f"current joint_pos: {obs['robot0_joint_pos'].round(4)}")
                        print(f"target_joints[0]:  {(obs['robot0_joint_pos'] + chunk[0, :7]).round(4)}")
                        print("=========================================")

                    for t in range(chunk_size):
                        delta         = chunk[t]
                        target_joints = obs["robot0_joint_pos"] + delta[:7]
                        gripper_cmd   = float(delta[7])
                        action        = np.concatenate([target_joints, [gripper_cmd]])
                        for _ in range(steps_per_action):
                            obs, _, _, _ = env.step(action)
                            if render:
                                env.render()
                        prev_action = delta.astype(np.float32)
                        state_buf.append(get_state(obs, gripper_cmd, prev_action))
                        image_buf.append(get_image(obs))
                        eef_pos_buf.append(obs["robot0_eef_pos"].copy())
                        eef_quat_buf.append(obs["robot0_eef_quat"].copy())
                        step += 1
                        if step >= max_steps or any(env.placement_success(b) for b in range(n_bins)):
                            break

                result = get_placed_bin(env)
                location_bins.append(result['location'])
                placement_bins.append(result['placement'])
                target_bins.append(target_bin)

                if metrics and eef_pos_buf:
                    from itps.envs.robosuite.trajectory_opt.geometric_features import (
                        BinXAlignment, BinYAlignment, ZTableDistance, GoalProgress
                    )
                    from itps.envs.robosuite.bin_placing import BinTableArena
                    positions = np.array(eef_pos_buf)
                    quats     = np.array(eef_quat_buf)
                    bx, by    = BinTableArena.BIN_XY[target_bin]
                    goal_pos  = obs_data['goal_pos'][idx]
                    feat_map  = {
                        'x_alignment':   BinXAlignment(x_bin=bx),
                        'y_alignment':   BinYAlignment(y_bin=by),
                        'z_table_dist':  ZTableDistance(table_z=0.8),
                        'goal_progress': GoalProgress(goal_pos=goal_pos),
                    }
                    for m in metrics:
                        if m in feat_map:
                            scores = feat_map[m](positions, quats)
                            feat_scores[m].append({'mean': float(scores.mean()), 'sum': float(scores.sum())})

            p = f'{method_label}/chunk{chunk_size}/{cond_label}'
            for m, ep_list in feat_scores.items():
                if ep_list:
                    all_metrics[f'{p}/{m}_mean'] = float(np.mean([e['mean'] for e in ep_list]))
                    all_metrics[f'{p}/{m}_sum']  = float(np.mean([e['sum']  for e in ep_list]))
            all_metrics[f'{p}/location_rate']        = float(np.mean([b is not None for b in location_bins]))
            all_metrics[f'{p}/placement_rate']       = float(np.mean([b is not None for b in placement_bins]))
            for i in range(n_bins):
                all_metrics[f'{p}/location_bin_{i}']  = sum(b == i for b in location_bins  if b is not None)
                all_metrics[f'{p}/placement_bin_{i}'] = sum(b == i for b in placement_bins if b is not None)
            if is_goal_cond:
                all_metrics[f'{p}/correct_location_rate']  = float(np.mean([b == t for b, t in zip(location_bins, target_bins)]))
                all_metrics[f'{p}/correct_placement_rate'] = float(np.mean([b == t for b, t in zip(placement_bins, target_bins)]))

    env.close()
    return {'aggregated': all_metrics}
