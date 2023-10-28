import re

import nltk
from nltk.translate.bleu_score import SmoothingFunction


def split_puncts(line):
    return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))


def nltk_sentence_bleu(hypothesis, reference):
    cc = SmoothingFunction()
    if len(hypothesis) == 1 and len(reference) == 1:
        if hypothesis == reference:
            return 1.
        else:
            return 0.
    return nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method4)


def f1_subtokens(pred, label):
    if len(pred) == 0:
        return 0.
    prec = len([p for p in pred if p in label]) / len(pred)
    recall = len([l for l in label if l in pred]) / len(label)
    if prec + recall == 0:
        return 0.
    else:
        return 2 * prec * recall / (prec + recall)


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def get_normalization(task, lang='python'):
    if task == 'code2text':
        return lambda s: split_puncts(s.lower().strip())
    elif task == 'codetrans':
        return lambda s: s.strip().split()
    elif task == 'func':
        if lang == 'python':
            return lambda s: [t.lower() for t in s.strip().split('_') if t != '']
        elif lang == 'java':
            return lambda s: [t.lower() for t in camel_case_split(s.strip()) if t != '']
