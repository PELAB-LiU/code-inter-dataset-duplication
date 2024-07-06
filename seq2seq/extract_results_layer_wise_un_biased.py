import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_results import get_avg_metrics

SEEDS_BIASED = [1]
SEEDS_UNBIASED = [11]
LAYERS = list(range(1, 11))
MODELS_BIASED = [f'random_biased_decoder_telly_{l}' for l in LAYERS]
MODELS_UNBIASED = [f'random_unbiased_decoder_telly_{l}' for l in LAYERS]

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()


results_dict = defaultdict(list)

for SEEDS, MODELS in zip([SEEDS_BIASED, SEEDS_UNBIASED], [MODELS_BIASED, MODELS_UNBIASED]):
    for seed in SEEDS:
        for model in tqdm(MODELS):
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
            results_dict['diff'].append(np.mean(dup) - np.mean(no_dup))
            results_dict['layer'].append(int(model.split('_')[-1]))
            model_kind = 'ff'
            results_dict['model_kind'].append(model_kind)

pd_df = pd.DataFrame.from_dict(results_dict)
pd_df.to_csv(args.output)
