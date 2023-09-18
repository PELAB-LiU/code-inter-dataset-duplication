import argparse
import os

import evaluate
import pandas as pd
from datasets import load_dataset


def main(args):
    results = pd.read_csv(os.path.join(args.data_folder, 'best_checkpoint', 'preds_labels_logits.csv'))
    dataset = load_dataset('antolin/bigclonebench_interduplication')
    results['is_duplicated'] = dataset['test']['is_duplicated']
    assert len(dataset['test']['is_duplicated']) == len(results)

    f1_metric = evaluate.load("f1")
    print(f1_metric.compute(predictions=results['pred_label'], references=results['true_label']))
    print(f1_metric.compute(predictions=list(results[results['is_duplicated']]['pred_label']),
                            references=list(results[results['is_duplicated']]['true_label'])))
    print(f1_metric.compute(predictions=list(results[~results['is_duplicated']]['pred_label']),
                            references=list(results[~results['is_duplicated']]['true_label'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    args = parser.parse_args()
    main(args)