import argparse

import pandas as pd
from check_differences import get_metric

biased_seeds = range(1, 11)
unbiased_seeds = range(11, 21)


def get_biase_performance(setting, path):
    results = []
    encoder = "encoder"
    for seed in biased_seeds:
        biased_pattern = f"{path}/biased/{encoder}/{seed}/{setting}/csn-small-biased-random-20-{setting}.bin.pkl"
        dup, non_dup, full = get_metric(biased_pattern)
        results.append((dup, non_dup, full))
    return results


def get_unbiase_performance(setting, path):
    unbiased_results = []
    encoder = "encoder"
    for seed in unbiased_seeds:
        unbiased_pattern = f"{path}/biased/{encoder}/{seed}/{setting}/csn-small-unbiased-random-20-{setting}.bin.pkl"
        dup, non_dup, full = get_metric(unbiased_pattern)
        unbiased_results.append((dup, non_dup, full))
    return unbiased_results


def get_performance(setting, is_bias, path):
    if is_bias:
        return get_biase_performance(setting, path)
    else:
        return get_unbiase_performance(setting, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    settings = ["ff", "lora", "prefix"]
    results = []

    for setting in settings:
        print(setting)
        biased_results = get_performance(setting, True, args.path)
        unbiased_results = get_performance(setting, False, args.path)

        for i, result in enumerate(biased_results):
            dup, non_dup, full = result
            results.append(
                {
                    "model": f"biased_{setting}",
                    "seed": biased_seeds[i],
                    "dup": dup,
                    "non_dup": non_dup,
                    "full": full,
                    "model_kind": setting,
                }
            )

        for i, result in enumerate(unbiased_results):
            dup, non_dup, full = result
            results.append(
                {
                    "model": f"unbiased_{setting}",
                    "seed": unbiased_seeds[i],
                    "dup": dup,
                    "non_dup": non_dup,
                    "full": full,
                    "model_kind": setting,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
