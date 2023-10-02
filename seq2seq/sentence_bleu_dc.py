import argparse
import math
import os

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from scipy.stats import ttest_ind, pearsonr

from bleu_code2text import normalize, splitPuncts
from train import f1_subtokens_python


def nltk_sentence_bleu(hypothesis, reference):
    cc = SmoothingFunction()
    if len(hypothesis) == 1 and len(reference) == 1:
        if hypothesis == reference:
            return 1
        else:
            return 0
    return nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method4)


def get_normalization(task):
    if task == 'code2text':
        return lambda s: normalize(
            splitPuncts(s.lower().strip()))
    elif task == 'codetrans':
        return lambda s: s.strip().split()
    elif task == 'func':
        return lambda s: s.strip()


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


# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


def confident_interval_cohen_d(d1, d2, Z=1.96):
    d = cohend(d1, d2)
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    spooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # df = min(n1 - 1, n2 - 1)
    n = 2 * (n1 * n2) / (n1 + n2)
    me = Z * (spooled / math.sqrt(n))
    return (d - me, d + me)


def main(args):
    normalization = get_normalization(args.task)
    references_full, predictions_full = read_data(args.data_folder, 'full', normalization)
    references_dup, predictions_dup = read_data(args.data_folder, 'dup', normalization)
    references_no_dup, predictions_no_dup = read_data(args.data_folder, 'no_dup', normalization)

    assert len(references_dup) == len(predictions_dup)
    assert len(references_full) == len(predictions_full)
    assert len(references_no_dup) == len(predictions_no_dup)

    if args.data_folder_control:
        references_dup_control, predictions_dup_control = read_data(args.data_folder_control, 'dup', normalization)
        references_no_dup_control, predictions_no_dup_control = read_data(args.data_folder_control, 'no_dup', normalization)

    if args.task == 'func':
        f1_full = [f1_subtokens_python(p, r) for p, r in zip(predictions_full, references_full)]
        f1_dup = [f1_subtokens_python(p, r) for p, r in zip(predictions_dup, references_dup)]
        f1_no_dup = [f1_subtokens_python(p, r) for p, r in zip(predictions_no_dup, references_no_dup)]

        print(f'F1 FULL: {np.mean(f1_full) * 100:.2f} +- {np.std(f1_full) * 100:.2f}')
        print(f'F1 NO DUP: {np.mean(f1_no_dup) * 100:.2f} +- {np.std(f1_no_dup) * 100:.2f}')
        print(f'F1 DUP: {np.mean(f1_dup) * 100:.2f} +- {np.std(f1_dup) * 100:.2f}')

        pval = ttest_ind(f1_no_dup, f1_dup, equal_var=False).pvalue
        print(f'p-value: {pval:.4f}')
        print(f'Cohen d: {cohend(f1_dup, f1_no_dup):.4f}')

        if args.data_folder_control:
            f1_dup_control = [f1_subtokens_python(p, r) for p, r in zip(predictions_dup_control, references_dup_control)]
            f1_no_dup_control = [f1_subtokens_python(p, r) for p, r in zip(predictions_no_dup_control, references_no_dup_control)]

            assert len(f1_dup) == len(f1_dup_control)
            improve_dup = [f1_dup[j] - f1_dup_control[j] for j, _ in enumerate(f1_dup)]
            assert len(f1_no_dup) == len(f1_no_dup_control)
            improve_no_dup = [f1_no_dup[j] - f1_no_dup_control[j] for j, _ in enumerate(f1_no_dup)]
            print(f'Improve dup {np.mean(improve_dup)*100:.2f} - Improve no dup {np.mean(improve_no_dup)*100:.2f}')
            pval = ttest_ind(improve_no_dup, improve_dup, equal_var=False).pvalue
            print(f'p-value: {pval:.4f}')
            print(f'Bias: {(np.mean(improve_dup) - np.mean(improve_no_dup)) * 100:.2f}')
    else:
        bleu_full = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_full, references_full)]
        bleu_dup = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_dup, references_dup)]
        bleu_no_dup = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_no_dup, references_no_dup)]

        print(f'BLEU FULL: {np.mean(bleu_full) * 100:.2f} +- {np.std(bleu_full) * 100:.2f}')
        print(f'BLEU NO DUP: {np.mean(bleu_no_dup) * 100:.2f} +- {np.std(bleu_no_dup) * 100:.2f}')
        print(f'BLEU DUP: {np.mean(bleu_dup) * 100:.2f} +- {np.std(bleu_dup) * 100:.2f}')

        pval = ttest_ind(bleu_no_dup, bleu_dup, equal_var=False).pvalue
        print(f'p-value: {pval:.4f}')
        print(f'Cohen d: {cohend(bleu_dup, bleu_no_dup):.4f}')
        print(f'Cohen d confidence: {confident_interval_cohen_d(bleu_dup, bleu_no_dup)}')

        if args.data_folder_control:
            bleu_dup_control = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_dup_control, references_dup_control)]
            bleu_no_dup_control = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_no_dup_control, references_no_dup_control)]
            assert len(bleu_dup_control) == len(bleu_dup)
            improve_dup = [bleu_dup[j] - bleu_dup_control[j] for j, _ in enumerate(bleu_dup)]
            assert len(bleu_no_dup_control) == len(bleu_no_dup)
            improve_no_dup = [bleu_no_dup[j] - bleu_no_dup_control[j] for j, _ in enumerate(bleu_no_dup)]
            print(f'Improve dup {np.mean(improve_dup) * 100:.2f} - Improve no dup {np.mean(improve_no_dup) * 100:.2f}')
            pval = ttest_ind(improve_no_dup, improve_dup, equal_var=False).pvalue
            print(f'p-value: {pval:.4f}')
            print(f'Bias: {(np.mean(improve_dup) - np.mean(improve_no_dup)) * 100:.2f}')

            # pearson
            p = pearsonr(bleu_dup_control + bleu_no_dup_control, bleu_dup + bleu_no_dup).statistic
            print(f'Pearson {p:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--task', choices=['codetrans', 'code2text', 'func'])
    parser.add_argument('--data_folder_control', type=str, default=None)
    parser.add_argument('--seed', default=123)
    args = parser.parse_args()
    main(args)
