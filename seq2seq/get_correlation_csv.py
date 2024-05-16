import argparse
from collections import defaultdict
from re import finditer

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import lizard

from analyze_results import get_avg_metrics

SEEDS = [123, 72, 93, 12345, 789]
MODELS = ['bart', 'rand66', 'rand33', 'rand63', 'codet5_ff', 't5v1', 'codet5large_ff',
          'codet5small_ff', 't5_fpfalse', 'codet5large_lora', 'codet5large_prefix']


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def normalize_code(tokens, lang):
    # return [t.lower() for t in tokens if t != '']
    new_tokens = []
    for t in tokens:
        if lang == "java":
            # camel case
            new_tokens += camel_case_split(t)
        elif lang == "python":
            if '_' in t:
                new_tokens += t.split('_')
    return [t.lower() for t in new_tokens if t != '']


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

# compute complexity metrics
dataset = load_dataset(args.dataset)["test"]
tokens_all = []
ccs = []
is_dup = []
nl_inter = []

for i in tqdm(range(len(dataset)), desc='Pred loop'):
    n_tokens = len(dataset[i]['tokens'])
    tokens_all.append(n_tokens)
    if args.dataset == 'antolin/codetrans_interduplication':
        analysis = lizard.analyze_file.analyze_source_code(f"test.{'py' if args.lang == 'python' else 'java'}",
                                                           dataset[i]['snippet'])
        ccs.append(analysis.function_list[0].cyclomatic_complexity)
    else:
        p = dataset[i]
        intersection = len(set(normalize_code(p['tokens'], args.lang)).intersection(set(p['nl'].lower().split()))) / len(
            set(p['nl'].lower().split()))
        nl_inter.append(intersection)

    is_dup.append(dataset[i]['is_duplicated'])

results_dict = defaultdict(list)

for seed in SEEDS:
    for model in tqdm(MODELS):
        folder = f'{args.folder}/seed_{seed}/{model}/best_checkpoint'
        try:
            full, dup, no_dup = get_avg_metrics(args.task, folder, args.lang)
        except:
            print(f'Cannot read {folder}')
            continue

        assert len(full) == len(tokens_all)

        results_dict['model'] += [model for _ in range(len(full))]
        results_dict['seed'] += [seed for _ in range(len(full))]
        results_dict['full'] += full
        results_dict['tokens'] += tokens_all
        if args.dataset == 'antolin/codetrans_interduplication':
            results_dict['cc'] += ccs
        else:
            results_dict['nl_inter'] += nl_inter
        results_dict['ids'] += [i for i in range(len(full))]
        results_dict['is_dup'] += is_dup

pd_df = pd.DataFrame.from_dict(results_dict)
pd_df.to_csv(args.output, index=False)
