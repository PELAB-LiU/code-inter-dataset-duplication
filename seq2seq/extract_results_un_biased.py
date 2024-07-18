import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_results import get_avg_metrics

SEEDS_BIASED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEEDS_UNBIASED = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
MODELS_BIASED = ['random_biased', 'random_biased_lora', 'random_biased_prefix']
MODELS_UNBIASED = ['random_unbiased', 'random_unbiased_lora', 'random_unbiased_prefix']

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()


results_dict = defaultdict(list)

for SEEDS, MODELS in zip([SEEDS_BIASED, SEEDS_UNBIASED], [MODELS_BIASED, MODELS_UNBIASED]):
    for seed in tqdm(SEEDS):
        for model in MODELS:
            folder = f'{args.folder}/seed_{seed}/{model}/best_checkpoint'
            try:
                full, dup, no_dup = get_avg_metrics(args.task, folder, args.lang)
            except:
                print(f'Cannot read {folder}')
                continue

            results_dict['model'].append(model)
            results_dict['seed'].append(seed)
            results_dict['dup'].append(np.mean(dup))
            results_dict['no_dup'].append(np.mean(no_dup))
            results_dict['full'].append(np.mean(full))
            model_kind = 'ff'
            if 'lora' in model:
                model_kind = 'lora'
            elif 'prefix' in model:
                model_kind = 'prefix'
            results_dict['model_kind'].append(model_kind)

pd_df = pd.DataFrame.from_dict(results_dict)
pd_df.to_csv(args.output, index=False)
