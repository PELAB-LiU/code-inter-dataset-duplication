import argparse
import os

import evaluate
import pandas as pd
from datasets import load_dataset


def main(args):
    results = pd.read_csv(os.path.join(args.data_folder, 'best_checkpoint', 'preds_labels.csv'))

    f1_metric = evaluate.load("f1")
    full = f1_metric.compute(predictions=results['pred_label'], references=results['true_label'])['f1']
    dup = f1_metric.compute(predictions=list(results[results['is_duplicated']]['pred_label']),
                            references=list(results[results['is_duplicated']]['true_label']))['f1']
    no_dup = f1_metric.compute(predictions=list(results[~results['is_duplicated']]['pred_label']),
                               references=list(results[~results['is_duplicated']]['true_label']))['f1']
    print('F1')
    print(f'Full {full * 100:.2f}')
    print(f'Dup {dup * 100:.2f}')
    print(f'No dup {no_dup * 100:.2f}')
    print(f'Diff {(dup - no_dup) * 100:.2f}')

    acc_metric = evaluate.load("accuracy")
    full = acc_metric.compute(predictions=results['pred_label'], references=results['true_label'])['accuracy']
    dup = acc_metric.compute(predictions=list(results[results['is_duplicated']]['pred_label']),
                             references=list(results[results['is_duplicated']]['true_label']))['accuracy']
    no_dup = acc_metric.compute(predictions=list(results[~results['is_duplicated']]['pred_label']),
                                references=list(results[~results['is_duplicated']]['true_label']))['accuracy']
    print('Acc')
    print(f'Full {full * 100:.2f}')
    print(f'Dup {dup * 100:.2f}')
    print(f'No dup {no_dup * 100:.2f}')
    print(f'Diff {(dup - no_dup) * 100:.2f}')

    print(f'Dup')
    dup_df = results[results['is_duplicated']]
    print("1", len(dup_df[dup_df["true_label"] == 1]) / len(dup_df))
    print("0", len(dup_df[dup_df["true_label"] == 0]) / len(dup_df))

    print(f'No dup')
    no_dup_df = results[~results['is_duplicated']]
    print("1", len(no_dup_df[no_dup_df["true_label"] == 1]) / len(no_dup_df))
    print("0", len(no_dup_df[no_dup_df["true_label"] == 0]) / len(no_dup_df))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    args = parser.parse_args()
    main(args)
