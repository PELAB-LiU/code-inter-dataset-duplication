import argparse
import os

import numpy as np
from scipy.stats import ttest_ind

from evaluation_metrics import get_normalization, nltk_sentence_bleu, f1_subtokens


def read_data(path, part, normalization):
    references_path = os.path.join(path, 'references' + f'_{part}.txt')
    predictions_path = os.path.join(path, 'predictions' + f'_{part}.txt')
    with open(references_path) as file:
        lines_references = file.readlines()
    references = [normalization(s) for s in lines_references]
    with open(predictions_path) as file:
        lines_predictions = file.readlines()
    predictions = [normalization(s) for s in lines_predictions]
    return references, predictions


def get_avg_metrics(task, data_folder, lang):
    normalization = get_normalization(task, lang)
    references_full, predictions_full = read_data(data_folder, 'full', normalization)
    references_dup, predictions_dup = read_data(data_folder, 'dup', normalization)
    references_no_dup, predictions_no_dup = read_data(data_folder, 'no_dup', normalization)
    eval_metric = f1_subtokens if task == "func" else nltk_sentence_bleu

    assert len(references_dup) == len(predictions_dup)
    assert len(references_full) == len(predictions_full)
    assert len(references_no_dup) == len(predictions_no_dup)

    full = [eval_metric(p, r) for p, r in zip(predictions_full, references_full)]
    dup = [eval_metric(p, r) for p, r in zip(predictions_dup, references_dup)]
    no_dup = [eval_metric(p, r) for p, r in zip(predictions_no_dup, references_no_dup)]
    return full, dup, no_dup


def main(args):
    full, dup, no_dup = get_avg_metrics(args.task, args.data_folder, args.lang)

    print(f"Task {args.task}")
    print(f'FULL: {np.mean(full) * 100:.2f} +- {np.std(full) * 100:.2f}')
    print(f'NO DUP: {np.mean(no_dup) * 100:.2f} +- {np.std(no_dup) * 100:.2f}')
    print(f'DUP: {np.mean(dup) * 100:.2f} +- {np.std(dup) * 100:.2f}')

    if args.data_folder_control:
        _, control_dup, control_no_dup = get_avg_metrics(args.task, args.data_folder_control)

        improvement_dup = [c1 - c2 for c1, c2 in zip(dup, control_dup)]
        improvement_no_dup = [c1 - c2 for c1, c2 in zip(no_dup, control_no_dup)]
        print(f'Improvement dup {np.mean(improvement_dup) * 100:.2f}')
        print(f'Improvement no dup {np.mean(improvement_no_dup) * 100:.2f}')

        pval = ttest_ind(improvement_no_dup, improvement_dup, equal_var=False).pvalue
        print(f'p-value: {pval:.4f}')
        print(f'Bias: {(np.mean(improvement_dup) - np.mean(improvement_no_dup)) * 100:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--task', choices=['codetrans', 'code2text', 'func'])
    parser.add_argument('--data_folder_control', type=str, default=None)
    parser.add_argument('--lang', type=str, default='python')
    args = parser.parse_args()
    main(args)
