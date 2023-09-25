import argparse
import os
import pickle
import random
import string

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from scipy.stats import ttest_ind, ranksums

from bleu_code2text import normalize, splitPuncts


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


def main(args):
    normalization = get_normalization(args.task)
    references_full, predictions_full = read_data(args.data_folder, 'full', normalization)
    references_dup, predictions_dup = read_data(args.data_folder, 'dup', normalization)
    references_no_dup, predictions_no_dup = read_data(args.data_folder, 'no_dup', normalization)
    random.seed(args.seed)
    combined_lists = list(zip(references_full, predictions_full))
    random.shuffle(combined_lists)
    references_ran, predictions_ran = zip(*combined_lists)
    references_ran, predictions_ran = references_ran[0:len(references_dup)], predictions_ran[0:len(predictions_dup)]
    assert len(references_dup) == len(predictions_dup)
    assert len(references_full) == len(predictions_full)
    assert len(references_no_dup) == len(predictions_no_dup)

    bleu_full = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_full, references_full)]
    bleu_dup = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_dup, references_dup)]
    bleu_no_dup = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_no_dup, references_no_dup)]
    bleu_rand = [nltk_sentence_bleu(p, r) for p, r in zip(predictions_ran, references_ran)]

    print(f'BLEU FULL: {np.mean(bleu_full) * 100:.2f} +- {np.std(bleu_full) * 100:.2f}')
    print(f'BLEU NO DUP: {np.mean(bleu_no_dup) * 100:.2f} +- {np.std(bleu_no_dup) * 100:.2f}')
    print(f'BLEU DUP: {np.mean(bleu_dup) * 100:.2f} +- {np.std(bleu_dup) * 100:.2f}')

    print(
        f'Length ref DUP: {np.mean([len(r) for r in references_dup]):.2f} +- {np.std([len(r) for r in references_dup]):.2f}')
    print(
        f'Length ref NO DUP: {np.mean([len(r) for r in references_no_dup]):.2f} +- {np.std([len(r) for r in references_no_dup]):.2f}')
    pval = ttest_ind([len(r) for r in references_no_dup], [len(r) for r in references_no_dup]).pvalue
    print(f'p-value ref length: {pval:.4f}')

    pval = ttest_ind(bleu_no_dup, bleu_dup, equal_var=False).pvalue
    pvalw = ranksums(bleu_no_dup, bleu_dup).pvalue
    print(f'p-value: {pval:.4f}')
    # print(f'p-valuew: {pvalw:.4f}')
    print(f'Cohen d: {cohend(bleu_dup, bleu_no_dup):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--task', choices=['codetrans', 'code2text'])
    parser.add_argument('--seed', default=123)
    args = parser.parse_args()
    main(args)
