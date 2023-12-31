import argparse
import os
from collections import defaultdict

import evaluate
import pandas as pd
from tqdm import tqdm

SEEDS = [123, 72, 93, 12345, 789]
MODELS = ['rand1', 'rand6', 'rand3', 'unixcoder', 'graphcodebert', 'codebert', 'roberta', 'mbert', 'bert']

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

results_dict = defaultdict(list)

for seed in SEEDS:
    for model in tqdm(MODELS):
        folder = f'{args.folder}/seed_{seed}/{model}/best_checkpoint'
        try:
            results = pd.read_csv(os.path.join(folder, 'preds_labels.csv'))
        except:
            print(f'Cannot read {folder}')
            continue

        f1_metric = evaluate.load("f1")
        full = f1_metric.compute(predictions=results['pred_label'], references=results['true_label'])['f1']
        dup = f1_metric.compute(predictions=list(results[results['is_duplicated']]['pred_label']),
                                references=list(results[results['is_duplicated']]['true_label']))['f1']
        no_dup = f1_metric.compute(predictions=list(results[~results['is_duplicated']]['pred_label']),
                                   references=list(results[~results['is_duplicated']]['true_label']))['f1']

        results_dict['model'].append(model)
        results_dict['seed'].append(seed)
        results_dict['dup'].append(dup)
        results_dict['no_dup'].append(no_dup)
        results_dict['full'].append(full)

pd_df = pd.DataFrame.from_dict(results_dict)
pd_df.to_csv(args.output)
