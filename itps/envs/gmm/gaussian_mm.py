#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Ground-truth definition of the GMM environment: named mixture specs, the post-hoc
preference utilities scored against them, and dataset generation.

A `GMMSpec` fully describes one GMM problem (weights, means, covariances). Specs are
registered by name in `GMM_SPECS` so a dataset, a checkpoint and an evaluation can all
refer to the same distribution without any of them hardcoding cluster positions or a
cluster count.

Preference is deliberately kept out of the spec. Training data is always drawn from the
spec's own (even) blend; a utility function `u(x)`, registered in `GMM_UTILITIES`, is
applied post-hoc to label preference pairs and to score a finetuned policy. Nothing in
the ground-truth density knows which points are preferred.
"""

import argparse
import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------- GMM density utilities -------

def mvn_pdf(X, mean, cov):
    """Multivariate normal density for 2D points X (N,2)."""
    X = np.atleast_2d(X)
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    diff = X - mean
    expo = np.einsum('...i,ij,...j->...', diff, inv, diff)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
    return norm * np.exp(-0.5 * expo)

def mixture_pdf(X, weights, means, covs):
    """Mixture density at points X given (weights, means, covs)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    total = np.zeros(X.shape[0])
    for wk, mk, Sk in zip(w, means, covs):
        total += wk * mvn_pdf(X, mk, Sk)
    return total

def sample_gmm(n_samples, weights, means, covs, seed=None):
    """
    Draw samples from a Gaussian Mixture Model.
    - weights: 1D array of shape (K,)
    - means: list/array of K elements, each (2,) for 2D
    - covs: list/array of K elements, each (2,2)
    - seed: seed of rng
    """
    rng = np.random.default_rng(seed)
    K = len(means)
    # choose components for each sample
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    comps = rng.choice(K, size=n_samples, p=w)

    # sample from each chosen component
    X = np.zeros((n_samples, 2))
    for k in range(K):
        idx = np.where(comps == k)[0]
        if idx.size:
            X[idx] = rng.multivariate_normal(mean=means[k], cov=covs[k], size=idx.size)
    return X, comps

# ------- Named GMM specs -------

@dataclass(frozen=True, eq=False)
class GMMSpec:
    """One named 2D Gaussian mixture. `weights` need not be normalized."""

    name: str
    weights: np.ndarray   # (K,)
    means: list           # K arrays of shape (2,)
    covs: list            # K arrays of shape (2, 2)

    @property
    def n_clusters(self):
        return len(self.means)

    def normalized_weights(self):
        w = np.asarray(self.weights, dtype=float)
        return w / w.sum()

    def pdf(self, X, centers=None):
        """
        Ground-truth density at X.

        centers=None uses the spec's own weights (the blend the data is drawn from).
        Passing a list of component indices instead puts all the weight on those
        components, which is how the conditional setting scores each cluster
        separately.
        """
        if centers is None:
            weights = self.normalized_weights()
        else:
            weights = np.zeros(self.n_clusters)
            weights[list(centers)] = 1.0
        return mixture_pdf(X, weights, self.means, self.covs)

    def sample(self, n_samples, seed=None):
        """Draw (X, component_index) from the mixture."""
        return sample_gmm(n_samples, self.weights, self.means, self.covs, seed)

    def as_dict(self):
        """JSON-friendly description, for recording alongside a generated dataset."""
        return {
            "name": self.name,
            "weights": np.asarray(self.weights, dtype=float).tolist(),
            "means": [np.asarray(m, dtype=float).tolist() for m in self.means],
            "covs": [np.asarray(c, dtype=float).tolist() for c in self.covs],
        }


def _isotropic_covs(scale, k):
    return [np.array([[scale, 0.0], [0.0, scale]]) for _ in range(k)]


GMM_SPECS = {
    # The original three clusters. Kept exactly as-is so results produced before the
    # specs existed stay reproducible.
    "three_cluster": GMMSpec(
        name="three_cluster",
        weights=np.array([1.0, 1.0, 1.0]),
        means=[
            np.array([0.0, 0.0]),
            np.array([3.5, 1.5]),
            np.array([-2.5, 3.0]),
        ],
        covs=[
            np.array([[0.10, 0.0],
                      [0.0, 0.10]]),   # tighter version of [[1.0, 0.6],[0.6,1.2]]

            np.array([[0.10, 0.0],
                      [0.0, 0.10]]),  # tighter version of [[0.8,-0.3],[-0.3,0.5]]

            np.array([[0.10, 0.0],
                      [0.0, 0.10]]),    # tighter version of [[0.6,0],[0,0.9]]
        ],
    ),
    # Clusters strung along the x=y diagonal, perturbed slightly off it. Pairs with the
    # "diagonal" utility: preference then increases monotonically along the line, so the
    # middle cluster sits between the other two rather than being all-or-nothing.
    "cluster_line": GMMSpec(
        name="cluster_line",
        weights=np.array([1.0, 1.0, 1.0]),
        means=[
            np.array([-2.72, -3.28]),
            np.array([-0.28, 0.28]),
            np.array([3.28, 2.72]),
        ],
        covs=_isotropic_covs(0.10, 3),
    ),
}

DEFAULT_SPEC = "three_cluster"


def get_spec(name):
    """Look up a registered GMM spec by name."""
    if name not in GMM_SPECS:
        raise KeyError(f"Unknown GMM spec '{name}'. Available: {sorted(GMM_SPECS)}")
    return GMM_SPECS[name]

# ------- Post-hoc preference utilities -------
# u(x) -> (N,) scalars, higher = more preferred. Applied after sampling, so the training
# distribution is untouched by the preference.

_DIAGONAL_DIRECTION = np.array([1.0, 1.0]) / np.sqrt(2.0)


def diagonal_utility(X):
    """Preference increases along the x=y direction (up and to the right)."""
    return np.asarray(X, dtype=float) @ _DIAGONAL_DIRECTION


GMM_UTILITIES = {"diagonal": diagonal_utility}


def get_utility(name):
    """Look up a registered preference utility by name."""
    if name not in GMM_UTILITIES:
        raise KeyError(f"Unknown GMM utility '{name}'. Available: {sorted(GMM_UTILITIES)}")
    return GMM_UTILITIES[name]

# ------- Dataset generation -------

def gen_dataset(spec, N, seed):
    """Draw N samples from `spec` in both conditional and unconditional observation format."""
    X, comps = spec.sample(N, seed)

    return {
        'X': X,
        'comps': comps,
        'conditional_observation': np.hstack([comps[:, None], X]),
        'unconditional_observation': np.hstack([np.zeros((len(X), 1)), X]),
        'spec': spec,
    }


def gen_demo_dataset(spec, N, seed, cluster=0):
    """Generate exactly N samples from a single cluster, in unconditional format."""
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mean=spec.means[cluster], cov=spec.covs[cluster], size=N)
    return np.hstack([np.zeros((N, 1)), X])


def save_observations(observations, path, **meta):
    """
    Save an (N, 3) observation array as .npy with a JSON sidecar recording how it was
    generated (spec, seed, utility, ...).

    The dataloader reads the bare array (`common/datasets/utils.py`, the 'npy' branch),
    so provenance has to live beside the file rather than inside it.
    """
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    np.save(path, observations)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path

# ------- Plotting -------

def plot_samples(X, x_range=(-8,8), y_range=(-8,8)):
    plt.figure(figsize=(6, 5))
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.scatter(X[:, 0], X[:, 1], s=8, alpha=0.6, edgecolor='none')
    plt.title("Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

def plot_gmm_pdf(spec, x_range=(-8,8), y_range=(-8,8)):
    """Filled contours of the ground-truth mixture density, with component means marked."""
    xmin, xmax = x_range
    ymin, ymax = y_range
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 200),
        np.linspace(ymin, ymax, 200)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    zz = spec.pdf(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=20)
    plt.title(f"Mixture density ({spec.name})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.tight_layout()

    # Optionally, overlay component means
    for m in spec.means:
        plt.plot(m[0], m[1], marker='x', markersize=8)

    plt.show()

def visualize_samples_and_pdf(dataset):
    spec = dataset['spec']
    for i in range(spec.n_clusters):
        cluster_idxs = np.where(dataset["comps"] == i)[0]
        plot_samples(dataset['X'][cluster_idxs])
    plot_samples(dataset['X'])

    plot_gmm_pdf(spec)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gmm-spec", type=str, default=DEFAULT_SPEC, choices=sorted(GMM_SPECS),
                        help=f"Which registered GMM to sample from (default: {DEFAULT_SPEC})")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--demo", action="store_true", help="Generate finetuning demo dataset (single cluster, unconditional format)")
    parser.add_argument("--demo-cluster", type=int, default=0, help="Which cluster to use for the demo dataset (default: 0)")
    parser.add_argument("--demo-n", type=int, default=1000, help="Number of demo samples to generate (default: 1000)")
    parser.add_argument("--save-path", type=str, default=None, help="Directory to save generated datasets (default: no saving)")
    parser.add_argument("--viz", action="store_true", help="Show samples and density plots (blocks until closed)")
    args = parser.parse_args()

    spec = get_spec(args.gmm_spec)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_path) if args.save_path is not None else None

    if args.demo:
        demo_obs = gen_demo_dataset(spec, args.demo_n, args.seed, cluster=args.demo_cluster)
        if args.viz:
            plot_samples(demo_obs[:, 1:])
        if save_dir is not None:
            path = save_observations(
                demo_obs,
                save_dir / f"gmm_{spec.name}_demo_cluster{args.demo_cluster}_{args.demo_n}_{args.seed}_{timestamp}.npy",
                spec=spec.as_dict(), n_samples=args.demo_n, seed=args.seed,
                kind="demo", cluster=args.demo_cluster,
            )
            print(f"Saved demo dataset to {path}")
    else:
        dataset = gen_dataset(spec, args.n, args.seed)

        if args.viz:
            visualize_samples_and_pdf(dataset)

        if save_dir is not None:
            for kind in ("unconditional", "conditional"):
                path = save_observations(
                    dataset[f'{kind}_observation'],
                    save_dir / f"gmm_{spec.name}_{kind}_{args.n}_{args.seed}_{timestamp}.npy",
                    spec=spec.as_dict(), n_samples=args.n, seed=args.seed, kind=kind,
                )
                print(f"Saved {kind} dataset to {path}")
