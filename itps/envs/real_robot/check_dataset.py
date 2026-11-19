"""Check the real_robot dataset loader against the converted file, read independently. No model is built.

    python itps/envs/real_robot/check_dataset.py --root itps/data/real_robot/cup_overfit10.hdf5 \
        [--config real_robot/overfit/run_dp_visual_overfit10] [--n_items 2000]

Goes through the real config and make_dataset (so the factory wiring is tested too), then checks, for sampled
training items, against values read straight from the hdf5:
  * observation.state / images: the right frames (t-1, t), first frame repeated at an episode start;
  * actions: anchor + chunk-relative label == the recorded absolute target [q[k+1], width[k+1]] (round trip),
    and the window's first joint label is exactly 0;
  * action stats == min/max recomputed here from the raw arrays, over the windows the sampler draws;
  * every drawn label normalizes into [-1, 1] (what clip_sample assumes);
  * glitch check: per-joint window max vs 99.9th percentile.
"""
import argparse
import os

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from itps.common.datasets.chunk_relative import N_ANCHORED_DIMS, from_chunk_relative
from itps.common.datasets.factory import make_dataset
from itps.common.datasets.sampler import EpisodeAwareSampler

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs")


def raw_episodes(root, load_images=True):
    with h5py.File(root, "r") as f:
        cams = f.attrs["cameras"].split(",") if load_images else []
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        return cams, [
            {"q": f[f"data/{k}/obs/joint_pos"][:], "w": f[f"data/{k}/obs/gripper_width"][:],
             "img": {c: f[f"data/{k}/obs/{c}_image"][:] for c in cams}}
            for k in keys
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="real_robot/overfit/run_dp_visual_overfit10")
    parser.add_argument("--n_items", type=int, default=2000)
    parser.add_argument("--no_images", action="store_true",
                        help="skip loading images (the full file needs ~2x its image bytes in RAM, ~18 GB)")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=os.path.abspath(CONFIG_DIR), version_base=None):
        cfg = compose("default", overrides=["env=real_robot", f"policy={args.config}",
                                            f"dataset_root={os.path.abspath(args.root)}"])
    if args.no_images:
        from omegaconf import open_dict

        with open_dict(cfg):
            for group in (cfg.policy.input_shapes, cfg.policy.input_normalization_modes, cfg.training.delta_timestamps):
                for key in [k for k in group if k.startswith("observation.image")]:
                    del group[key]
    ds = make_dataset(cfg)
    drop = cfg.training.drop_n_last_frames
    policy_keys = set(cfg.policy.input_shapes) | set(cfg.policy.output_shapes)
    missing = policy_keys - set(ds.stats)
    assert not missing, f"policy inputs/outputs without dataset stats: {missing}"
    assert policy_keys <= set(cfg.training.delta_timestamps), "policy keys without a delta_timestamps window"
    offsets = ds._chunk_offsets("action")
    obs_offsets = ds._chunk_offsets("observation.state")
    cams, eps = raw_episodes(args.root, load_images=not args.no_images)
    print(f"dataset: {ds.num_episodes} episodes, {len(ds)} rows, cameras {cams}, action offsets "
          f"{offsets[0]}..{offsets[-1]}, drop_n_last_frames {drop}")

    # Row -> (episode, frame) and the sampler's anchors.
    ep_of_row = np.concatenate([np.full(len(e["q"]) - 1, i) for i, e in enumerate(eps)])
    frame_of_row = np.concatenate([np.arange(len(e["q"]) - 1) for e in eps])
    anchors = list(EpisodeAwareSampler(ds.episode_data_index, drop_n_last_frames=drop, shuffle=False))
    assert len(ds) == len(ep_of_row), "row count differs from the file"

    # Independent stats over the sampler's windows.
    labels = []
    for e in eps:
        n = len(e["q"]) - 1
        for t in range(n - drop):
            k = t + offsets
            k = k[(k >= 0) & (k < n)]
            lab = np.concatenate([e["q"][k + 1] - e["q"][t], e["w"][k + 1, None]], axis=1)
            labels.append(lab)
    labels = np.concatenate(labels)
    assert np.allclose(ds.stats["action"]["min"].numpy(), labels.min(0), atol=1e-6), "action min differs"
    assert np.allclose(ds.stats["action"]["max"].numpy(), labels.max(0), atol=1e-6), "action max differs"
    assert len(anchors) == sum(len(e["q"]) - 1 - drop for e in eps), "sampler anchors != windows in the stats"
    print(f"[stats] action min/max match an independent recomputation over {len(anchors)} windows")
    for j in range(N_ANCHORED_DIMS):
        hi = np.abs(labels[:, j]).max(); p = np.percentile(np.abs(labels[:, j]), 99.9)
        print(f"        j{j}: range [{labels[:, j].min():+.3f}, {labels[:, j].max():+.3f}]  max/p99.9 {hi / p:.2f}")
    print(f"        gripper width: [{labels[:, 7].min():.4f}, {labels[:, 7].max():.4f}]")

    # Item-level checks on a sample of the sampler's anchors (always including episode starts and ends).
    rng = np.random.default_rng(0)
    starts = ds.episode_data_index["from"].tolist()
    picks = sorted(set(rng.choice(anchors, size=min(args.n_items, len(anchors)), replace=False).tolist())
                   | set(starts) | {a for a in anchors if a + 1 not in set(anchors)})
    a_min, a_max = ds.stats["action"]["min"], ds.stats["action"]["max"]
    for row in picks:
        item = ds[row]
        e, t = eps[ep_of_row[row]], frame_of_row[row]
        n = len(e["q"]) - 1
        obs_frames = np.clip(t + obs_offsets, 0, n - 1)
        exp_state = np.concatenate([e["q"][obs_frames], e["w"][obs_frames, None]], axis=1)
        assert np.allclose(item["observation.state"].numpy(), exp_state, atol=1e-6), f"row {row}: state"
        for c in cams:
            exp_img = e["img"][c][obs_frames].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            got = item[f"observation.image.{c}"]
            assert got.shape == (len(obs_offsets), 3, *exp_img.shape[2:]), f"row {row}: {c} shape {got.shape}"
            assert np.array_equal(got.numpy(), exp_img), f"row {row}: {c} pixels"
        k = t + offsets
        valid = (k >= 0) & (k < n)
        assert torch.equal(item["action_is_pad"], torch.from_numpy(~valid)), f"row {row}: padding mask"
        absolute = from_chunk_relative(item["action"], item["observation.state"][-1]).numpy()
        exp_abs = np.concatenate([e["q"][k[valid] + 1], e["w"][k[valid] + 1, None]], axis=1)
        assert np.allclose(absolute[valid], exp_abs, atol=1e-5), f"row {row}: anchor + label != recorded target"
        if valid[0] and offsets[0] == -1:
            assert np.abs(item["action"][0, :N_ANCHORED_DIMS].numpy()).max() < 1e-5, f"row {row}: first label != 0"
        lab = item["action"][torch.from_numpy(valid)]
        normalized = (lab - a_min) / (a_max - a_min) * 2 - 1
        assert normalized.min() >= -1 - 1e-5 and normalized.max() <= 1 + 1e-5, f"row {row}: label outside [-1, 1]"
    print(f"[items] {len(picks)} items (incl. every episode's first and last anchor): states, "
          f"{'frames of ' + ', '.join(cams) if cams else 'NO images (--no_images)'}, padding mask, "
          f"anchor + label == recorded target, first label 0, normalized in [-1, 1]")
    print("OK")


if __name__ == "__main__":
    main()
