
import copy 
import argparse
from gaussian_mm import * 

def generate_preference_data(dataset, pref_mode):

    pref_dict = copy.deepcopy(dataset)
    pref_dict['pref_mode']=pref_mode

    pos_idxs = np.where(dataset["comps"]==pref_mode)[0]

    # Get subset of negative indices to match
    neg_idxs = np.where(dataset["comps"]!=pref_mode)[0] 
    neg_idxs_sub = np.random.choice(neg_idxs, size=len(pos_idxs), replace=False)

    pref_dict['positive_observation'] = np.hstack(np.zeros((len(pos_idxs),1)), [pref_dict['X'][pos_idxs]])
    pref_dict['negative_observation'] = np.hstack(np.zeros((len(pos_idxs),1)), [pref_dict['X'][neg_idxs_sub]])

    return pref_dict 
   
def plot_pref_dataset(pref_dataset):

    # Don't plot observation (first element of 2nd dimension)
    plot_samples(pref_dataset['positive_observation'][:,1:])
    plot_samples(pref_dataset['negative_observation'][:,1:])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--ref-dir",
        help=(
            "Set of reference samples. If not provided, samples are regenerated "
        ),
    )
    parser.add_argument(
        "--pref-cluster",
        default=0,
        help=(
            "Preferred cluster. If not provided, chooses cluster 0."
        ),
    )

    args= parser.parse_args()

    # IF FILE NOT PASSED IN: 
    if args.ref_dir is None:
        N=1000
        seed=42
        dataset = gen_dataset(N, seed)

    else:
        dataset = np.load(args.ref_dir)

    # Filter dataset and visualize
    pref_dataset = generate_preference_data(dataset, args.pref_cluster)
    plot_pref_dataset(pref_dataset)

    save_dir = "data/"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pref_pos_file = f"gmm_unconditional_pref_cluster_{args.pref_cluster}_positive_dataset_{timestamp}.npy"
    pref_neg_file = f"gmm_unconditional_pref_cluster_{args.pref_cluster}_negative_dataset_{timestamp}.npy"
    np.save(save_dir+pref_pos_file, pref_dataset['positive_observation'])
    np.save(save_dir+pref_neg_file, pref_dataset['negative_observation'])
