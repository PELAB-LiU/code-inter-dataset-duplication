import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def importance_sampling(score_dup, score_nodup, ids_nodup, seed=123):
    np.random.seed(seed)

    pdf_dup = gaussian_kde(score_dup)
    pdf_nodup = gaussian_kde(score_nodup)

    # Compute importance weights
    weights = pdf_dup(score_nodup) / pdf_nodup(score_nodup)

    # Normalize weights
    weights /= np.sum(weights)

    # Resample with replacement based on weights

    new_samples_group1 = np.random.choice(np.arange(len(score_nodup)),
                                          size=len(score_dup),
                                          replace=True, p=weights)

    return ids_nodup[new_samples_group1], score_nodup[new_samples_group1]


def main(args):
    control_model = pd.read_csv(args.csv)

    print(f"Performance dup: {np.mean(control_model[control_model['is_dup'] == 1.]['performance'])} "
          f"vs Performance no dup: {np.mean(control_model[control_model['is_dup'] == 0.]['performance'])}")
    print(f"Performance dup: {np.std(control_model[control_model['is_dup'] == 1.]['performance'])} "
          f"vs Performance no dup: {np.std(control_model[control_model['is_dup'] == 0.]['performance'])}")

    ids_nodup = control_model[control_model['is_dup'] == 0.]['id'].to_numpy()
    score_dup = control_model[control_model['is_dup'] == 1.]['performance'].to_numpy()
    score_nodup = control_model[control_model['is_dup'] == 0.]['performance'].to_numpy()
    new_ids, new_scores = importance_sampling(score_dup, score_nodup, ids_nodup, seed=args.seed)

    # assert len(new_scores) == len(control_model[control_model['is_dup'] == 1.]['performance'])

    print("After sampling")
    print(f"Performance dup: {np.mean(control_model[control_model['is_dup'] == 1.]['performance'])} "
          f"vs Performance no dup: {np.mean(new_scores)}")
    print(f"Performance dup: {np.std(control_model[control_model['is_dup'] == 1.]['performance'])} "
          f"vs Performance no dup: {np.std(new_scores)}")

    assert control_model[control_model['id'].isin(new_ids.tolist())]['is_dup'].sum() == 0.

    with open(f'{args.output}', 'w') as f:
        json.dump(new_ids.tolist() + control_model[control_model['is_dup'] == 1.]['id'].to_numpy().tolist(), f)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args)
