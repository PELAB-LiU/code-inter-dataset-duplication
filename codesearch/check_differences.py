import argparse
import pickle

import numpy as np
from scipy.stats import ttest_ind


def get_metric(data):
    with open(data, "rb") as handle:
        data = pickle.load(handle)

    dup = np.mean(data[1])
    non_dup = np.mean(data[0])
    full = np.mean(data[0] + data[1])

    return dup, non_dup, full


def main(data, data_control):
    with open(data, "rb") as handle:
        data = pickle.load(handle)
    with open(data_control, "rb") as handle:
        data_control = pickle.load(handle)
    diff_dup = [r - r_control for r, r_control in zip(data[1], data_control[1])]
    diff_no_dup = [r - r_control for r, r_control in zip(data[0], data_control[0])]

    # print(f'Mean diff dup: {np.mean(diff_dup):.4f}')
    # print(f'Mean diff no dup: {np.mean(diff_no_dup):.4f}')
    # print(f'Bias: {np.mean(diff_dup) - np.mean(diff_no_dup):.4f}')
    # pval = ttest_ind(diff_no_dup, diff_dup, equal_var=False).pvalue
    # print(f'p-value: {pval:.4f}')

    return np.mean(diff_dup) - np.mean(diff_no_dup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--data_control", type=str, default=None)
    args = parser.parse_args()
    main(args.data, args.data_control)
