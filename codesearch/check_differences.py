import argparse
import pickle

import numpy as np
from scipy.stats import ttest_ind


def main(args):
    with open(args.data, 'rb') as handle:
        data = pickle.load(handle)
    with open(args.data_control, 'rb') as handle:
        data_control = pickle.load(handle)

    diff_dup = [r - r_control for r, r_control in zip(data[1], data_control[1])]
    diff_no_dup = [r - r_control for r, r_control in zip(data[0], data_control[0])]

    print(f'Mean diff dup: {np.mean(diff_dup):.4f}')
    print(f'Mean diff no dup: {np.mean(diff_no_dup):.4f}')
    print(f'Bias: {np.mean(diff_dup) - np.mean(diff_no_dup):.4f}')
    pval = ttest_ind(diff_no_dup, diff_dup, equal_var=False).pvalue
    print(f'p-value: {pval:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--data_control', type=str, default=None)
    args = parser.parse_args()
    main(args)