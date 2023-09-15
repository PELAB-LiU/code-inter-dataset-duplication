import argparse
import json
from datasets import load_dataset, DatasetDict


def main(args):
    # load dup_ids
    with open(args.inter, 'r') as json_file:
        dup_ids = json.load(json_file)

    # load representatives
    with open(args.rep, 'r') as json_file:
        representatives = json.load(json_file)

    # load dataset
    full_dataset = load_dataset("json", data_files=args.data)["train"].filter(
        lambda x: x['id_within_dataset'] in representatives)
    is_duplicated = [True if i in dup_ids else False for i in full_dataset['id_within_dataset']]
    full_dataset = full_dataset.add_column('is_duplicated', is_duplicated)
    train_dataset = full_dataset.filter(lambda example: example['split_within_dataset'] == 'train')
    test_dataset = full_dataset.filter(lambda example: example['split_within_dataset'] == 'test')
    valid_dataset = full_dataset.filter(lambda example: example['split_within_dataset'] == 'valid')
    new_dataset = DatasetDict({"train": train_dataset,
                               "test": test_dataset,
                               "valid": valid_dataset})
    print(new_dataset)
    new_dataset.push_to_hub(args.hf_dataset, private=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--inter', type=str, required=True)
    parser.add_argument('--rep', type=str, required=True)
    parser.add_argument('--hf_dataset', type=str, required=True)
    args = parser.parse_args()
    main(args)
