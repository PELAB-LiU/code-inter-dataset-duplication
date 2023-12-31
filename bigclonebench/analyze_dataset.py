import math

import datasets
import numpy as np
from dpu_utils.codeutils import get_language_keywords
from dpu_utils.codeutils.deduplication import DuplicateDetector
from scipy.stats import ttest_ind
from tqdm import tqdm

dataset = datasets.load_dataset('antolin/bigclonebench_interduplication')['test']


# dataset = datasets.concatenate_datasets([dataset['train'], dataset['valid'], dataset['test']])

def cohen_d(group1, group2):
    """Calculate Cohen's d for two groups."""
    mean_diff = abs(np.mean(group1) - np.mean(group2))
    pooled_std = math.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    cohen_d = mean_diff / pooled_std
    return cohen_d


dup_group_jaccard_set = {
    "positive": [],
    "negative": []
}
no_dup_group_jaccard_set = {
    "positive": [],
    "negative": []
}
dup_group_jaccard_multiset = {
    "positive": [],
    "negative": []
}
no_dup_group_jaccard_multiset = {
    "positive": [],
    "negative": []
}


def jaccard_multiset_func(multiset_tokens1, multiset_tokens2):
    m1 = {t: multiset_tokens1.count(t) for t in multiset_tokens1}
    m2 = {t: multiset_tokens2.count(t) for t in multiset_tokens2}
    intersection = 0
    for t in m1:
        if t in m2:
            intersection += min(m1[t], m2[t])
    union = 0
    for t in m1:
        if t in m2:
            union += max(m1[t], m2[t])
        else:
            union += m1[t]
    for t in m2:
        if t not in m1:
            union += m2[t]

    return intersection / union


for row in tqdm(dataset, desc="Analyzing dataset", total=len(dataset)):
    is_dup = row['is_duplicated']
    label = row['label']
    set_tokens1 = set([t for t in set(row['tokens1']) if DuplicateDetector.IDENTIFIER_REGEX.match(t)
                       and (t not in get_language_keywords("java"))])
    set_tokens2 = set([t for t in set(row['tokens2']) if DuplicateDetector.IDENTIFIER_REGEX.match(t)
                       and (t not in get_language_keywords("java"))])
    jaccard_set = len(set_tokens1.intersection(set_tokens2)) / len(set_tokens1.union(set_tokens2))

    multiset_tokens1 = [t for t in row['tokens1'] if DuplicateDetector.IDENTIFIER_REGEX.match(t)
                        and (t not in get_language_keywords("java"))]
    multiset_tokens2 = [t for t in row['tokens2'] if DuplicateDetector.IDENTIFIER_REGEX.match(t)
                        and (t not in get_language_keywords("java"))]
    jaccard_multiset = jaccard_multiset_func(multiset_tokens1, multiset_tokens2)
    if is_dup:
        if label:
            dup_group_jaccard_set['positive'].append(jaccard_set)
            dup_group_jaccard_multiset['positive'].append(jaccard_multiset)
        else:
            dup_group_jaccard_set['negative'].append(jaccard_set)
            dup_group_jaccard_multiset['negative'].append(jaccard_multiset)
    else:
        if label:
            no_dup_group_jaccard_set['positive'].append(jaccard_set)
            no_dup_group_jaccard_multiset['positive'].append(jaccard_multiset)
        else:
            no_dup_group_jaccard_set['negative'].append(jaccard_set)
            no_dup_group_jaccard_multiset['negative'].append(jaccard_multiset)

print("Duplication group")
print(f'Positive {np.mean(dup_group_jaccard_set["positive"]):.4f} +- {np.std(dup_group_jaccard_set["positive"]):.4f}')
print(f'Negative {np.mean(dup_group_jaccard_set["negative"]):.4f} +- {np.std(dup_group_jaccard_set["negative"]):.4f}')
print(f'Positive multiset {np.mean(dup_group_jaccard_multiset["positive"]):.4f} +- '
      f'{np.std(dup_group_jaccard_multiset["positive"]):.4f}')
print(f'Negative multiset {np.mean(dup_group_jaccard_multiset["negative"]):.4f} +- '
      f'{np.std(dup_group_jaccard_multiset["negative"]):.4f}')

print("No duplication group")
print(f'Positive {np.mean(no_dup_group_jaccard_set["positive"]):.4f} +- '
      f'{np.std(no_dup_group_jaccard_set["positive"]):.4f}')
print(f'Negative {np.mean(no_dup_group_jaccard_set["negative"]):.4f} +- '
      f'{np.std(no_dup_group_jaccard_set["negative"]):.4f}')
print(f'Positive multiset {np.mean(no_dup_group_jaccard_multiset["positive"]):.4f} +- '
      f'{np.std(no_dup_group_jaccard_multiset["positive"]):.4f}')
print(f'Negative multiset {np.mean(no_dup_group_jaccard_multiset["negative"]):.4f} +- '
      f'{np.std(no_dup_group_jaccard_multiset["negative"]):.4f}')

print("Dup vs no dup, t-test and size effect")
print("Positive")
print(f'p-val: '
      f'{ttest_ind(dup_group_jaccard_set["positive"], no_dup_group_jaccard_set["positive"]).pvalue:.4f}')
print(f'Effect size: {cohen_d(dup_group_jaccard_set["positive"], no_dup_group_jaccard_set["positive"]):.2f}')
print("Negative")
print(f'p-val: '
      f'{ttest_ind(dup_group_jaccard_set["negative"], no_dup_group_jaccard_set["negative"]).pvalue:.4f}')
print(f'Effect size: {cohen_d(dup_group_jaccard_set["negative"], no_dup_group_jaccard_set["negative"]):.2f}')

print("Dup vs no dup, t-test and size effect, multiset")
print("Positive")
print(f'p-val: '
      f'{ttest_ind(dup_group_jaccard_multiset["positive"], no_dup_group_jaccard_multiset["positive"]).pvalue:.4f}')
print(f'Effect size: {cohen_d(dup_group_jaccard_multiset["positive"], no_dup_group_jaccard_multiset["positive"]):.2f}')
print("Negative")
print(f'p-val: '
      f'{ttest_ind(dup_group_jaccard_multiset["negative"], no_dup_group_jaccard_multiset["negative"]).pvalue:.4f}')
print(f'Effect size: {cohen_d(dup_group_jaccard_multiset["negative"], no_dup_group_jaccard_multiset["negative"]):.2f}')
