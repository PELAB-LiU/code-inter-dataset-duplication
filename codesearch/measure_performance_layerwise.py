import argparse

import pandas as pd
from check_differences import get_metric

layerwise_seeds = range(1, 6)
layers = range(1, 12)


def load_layerwise(layer, seed, is_biased, path):
    if is_biased:
        pattern = (
            f"{path}/biased/layerwise/{seed}/csn-small-biased-random-20-{layer}.bin.pkl"
        )
    else:
        pattern = f"{path}/unbiased/layerwise/{seed}/csn-small-unbiased-random-20-{layer}.bin.pkl"
    return get_metric(pattern)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    results = []

    for seed in layerwise_seeds:
        for layer in layers:
            dup, non_dup, full = load_layerwise(layer, seed, True, args.path)
            results.append(
                {
                    "model": f"biased_layerwise_{layer}",
                    "seed": seed,
                    "dup": dup,
                    "non_dup": non_dup,
                    "full": full,
                    "layer": layer,
                }
            )

            dup, non_dup, full = load_layerwise(layer, seed, False, args.path)
            results.append(
                {
                    "model": f"unbiased_layerwise_{layer}",
                    "seed": seed,
                    "dup": dup,
                    "non_dup": non_dup,
                    "full": full,
                    "layer": layer,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
