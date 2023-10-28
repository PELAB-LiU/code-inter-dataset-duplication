from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_results import get_avg_metrics

FOLDER = 'codetrans'
SEEDS = [72, 123, 93]
MODELS = ['t5', 'bart', 'rand66', 'rand33', 'rand63', 'codet5_lora', 'codet5_ff', 'codet5_prefix', 't5v1']
TASKS = 'codetrans'
OUTPUT = 'results_codetrans.csv'
LANG = 'java'

results_dict = defaultdict(list)

for seed in SEEDS:
    for model in tqdm(MODELS):
        folder = f'{FOLDER}/seed_{seed}/{model}/best_checkpoint'
        full, dup, no_dup = get_avg_metrics(TASKS, folder, LANG)

        results_dict['model'].append(model)
        results_dict['seed'].append(seed)
        results_dict['dup'].append(np.mean(dup))
        results_dict['no_dup'].append(np.mean(no_dup))
        results_dict['full'].append(np.mean(full))

pd_df = pd.DataFrame.from_dict(results_dict)
pd_df.to_csv(OUTPUT)
