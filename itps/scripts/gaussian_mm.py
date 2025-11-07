import numpy as np
import datetime
import matplotlib.pyplot as plt

# ------- GMM utilities -------

def sample_gmm(n_samples, weights, means, covs, seed=None):
    """
    Draw samples from a Gaussian Mixture Model.
    - weights: 1D array of shape (K,)
    - means: list/array of K elements, each (2,) for 2D
    - covs: list/array of K elements, each (2,2)
    - seed: seed of rng
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()  # ensure normalization
    K = len(weights)

    # choose components for each sample
    comps = rng.choice(K, size=n_samples, p=weights)

    # sample from each chosen component
    X = np.zeros((n_samples, 2))
    for k in range(K):
        idx = np.where(comps == k)[0]
        if idx.size:
            X[idx] = rng.multivariate_normal(mean=means[k], cov=covs[k], size=idx.size)
    return X, comps

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

# ------- Generate samples -------
def gen_samples(weights, means, covs, N, seed):
    X, comps = sample_gmm(N, weights, means, covs, seed)
    return X, comps

def get_weights():
    return  np.array([0.45, 0.35, 0.20])

def get_means():
    return [
        np.array([0.0, 0.0]),
        np.array([3.5, 1.5]),
        np.array([-2.5, 3.0]),
    ]

def get_covs():
    return [
        np.array([[1.0, 0.6],
                [0.6, 1.2]]),
        np.array([[0.8, -0.3],
                [-0.3, 0.5]]),
        np.array([[0.6, 0.0],
                [0.0, 0.9]]),
    ]


def gen_dataset(N, seed):

    # ------- Example configuration -------
    # 3-component 2D GMM
    weights = get_weights()
    means = get_means()
    covs = get_covs()
    X, comps = gen_samples(weights, means, covs, N, seed)

    data_dict ={
        'X': X,
        'comps': comps,
        'conditional_observation': np.hstack([comps[:, None], X]),
        'unconditional_observation':np.hstack([np.zeros((len(X),1)), X]),
        'weights': weights,
        'means': means,
        'covs': covs
    }

    return data_dict

# ------- Plot sampled points -------
def plot_samples(X, x_range=(-8,8), y_range=(-8,8)):
    plt.figure(figsize=(6, 5))
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.scatter(X[:, 0], X[:, 1], s=8, alpha=0.6, edgecolor='none')
    plt.title("Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ------- Plot mixture density (contours) -------
# grid over data range with padding
def plot_gmm_pdf(weights, means, covs, x_range=(-8,8), y_range=(-8,8)):
    #pad = 2.0
    xmin, xmax = x_range #X.min(axis=0) - pad
    ymin, ymax = y_range #X.max(axis=0) + pad
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 200),
        np.linspace(ymin, ymax, 200)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    zz = mixture_pdf(grid, weights, means, covs).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=20)
    plt.title("Mixture Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.tight_layout()

    # Optionally, overlay component means
    for m in means:
        plt.plot(m[0], m[1], marker='x', markersize=8)

    plt.show()

def visualize_samples_and_pdf(dataset):
    for i in range(3):
        cluster_idxs = np.where(dataset["comps"]==i)[0]
        plot_samples(dataset['X'][cluster_idxs])
    plot_samples(dataset['X'])

    plot_gmm_pdf(weights=dataset['weights'],
                means=dataset['means'],
                covs=dataset['covs'],
                )

if __name__ == "__main__":

    N=1000
    seed=42
    dataset = gen_dataset(N, seed)

    # -- Comment out to generate visualizations -- 
    visualize_samples_and_pdf(dataset)

    #save_dir = "data/"
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #unconditional_file = f"gmm_unconditional_{N}_{seed}_{timestamp}.npy"
    #conditional_file = f"gmm_conditional_{N}_{seed}_{timestamp}.npy"

    #np.save(save_dir+unconditional_file, dataset['unconditional_observation'])
    #np.save(save_dir+conditional_file, dataset['conditional_observation'])
