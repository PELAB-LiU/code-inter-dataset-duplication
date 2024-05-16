import argparse
import json
from collections import defaultdict

import pandas as pd
from datasets import load_dataset

from analyze_results import get_avg_metrics

SEEDS = [123, 72, 93, 12345, 789]


def main(args):
    # get duplication
    dataset = load_dataset(args.dataset)["test"]
    is_dup = [dataset[i]['is_duplicated'] for i in range(len(dataset))]
    ids = [i for i in range(len(dataset))]

    results_dict = defaultdict(list)

    for seed in SEEDS:
        folder = f'{args.folder}/seed_{seed}/{args.model}/best_checkpoint'
        try:
            full, dup, no_dup = get_avg_metrics(args.task, folder, args.lang)
        except:
            print(f'Cannot read {folder}')
            continue
        results_dict['performance'] += full
        results_dict['id'] += ids
        results_dict['is_dup'] += is_dup

    pd_df = pd.DataFrame.from_dict(results_dict)
    results = pd_df.groupby(['id']).mean().reset_index()
    results['model'] = args.model
    results.to_csv(f'{args.output}', index=False)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args)
