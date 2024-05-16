import argparse
import json
import os

import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm

from register_in_db import load_dataset as ld


def is_utf8(x):
    try:
        bytes(x, 'utf-8')
        return True
    except:
        return False


def main(args):
    with open(os.path.join(args.target_dataset, f'dups_java.json'), 'r') as json_file:
        dup_ids = json.load(json_file)
    with open(os.path.join(args.target_dataset, f'dups_python.json'), 'r') as json_file:
        dup_ids += json.load(json_file)
    dup_ids = set(dup_ids)

    if args.first_stage:
        # load dataset
        full_dataset = [x for x in ld(args.target_dataset + "/data.jsonl")]
        full_dataset = [x for x in tqdm(full_dataset) if all(is_utf8(t) for t in x["tokens"])]
        full_dataset = [x for x in tqdm(full_dataset) if x["nl"].strip() != ""]

        with open(os.path.join(args.target_dataset, f'data_new.jsonl'), 'w') as file:
            for x in tqdm(full_dataset):
                json_string = json.dumps(x)
                file.write(json_string + '\n')
    else:

        full_dataset = load_dataset("json", data_files=args.target_dataset + "/data_new.jsonl")["train"]

        dup_dataset = full_dataset.filter(
            lambda x: x['id_within_dataset'] in dup_ids, num_proc=5, load_from_cache_file=False)
        dup_dataset = dup_dataset.add_column('is_duplicated', [True for _ in range(len(dup_dataset))])

        nodup_dataset = full_dataset.filter(
            lambda x: x['id_within_dataset'] not in dup_ids, num_proc=5, load_from_cache_file=False)
        nodup_dataset = nodup_dataset.add_column('is_duplicated', [False for _ in range(len(nodup_dataset))])

        ### java and python

        dup_dataset_java = dup_dataset.filter(lambda x: x['language'] == 'java', num_proc=5)
        dup_dataset_python = dup_dataset.filter(lambda x: x['language'] == 'python', num_proc=5)

        nodup_dataset_java = nodup_dataset.filter(lambda x: x['language'] == 'java', num_proc=5)
        nodup_dataset_python = nodup_dataset.filter(lambda x: x['language'] == 'python', num_proc=5)

        ### base dataset
        nodup_dataset_java = nodup_dataset_java.shuffle(seed=123)
        nodup_dataset_python = nodup_dataset_python.shuffle(seed=123)
        base_dataset_java = nodup_dataset_java.select(range(0, args.samples // 2))
        base_dataset_python = nodup_dataset_python.select(range(0, args.samples // 2))
        base_dataset = concatenate_datasets([base_dataset_python, base_dataset_java])

        ### biased dataset
        biased_dataset = concatenate_datasets([base_dataset, dup_dataset])

        ### unbiased dataset
        java_part = nodup_dataset_java.select(range(args.samples // 2, (args.samples // 2) + len(dup_dataset_java)))
        python_part = nodup_dataset_python.select(range(args.samples // 2, (args.samples // 2) + len(dup_dataset_python)))

        unbiased_dataset = concatenate_datasets([base_dataset, java_part, python_part])

        assert len(unbiased_dataset.filter(lambda x: x['language'] == 'python', num_proc=5)) == len(
            biased_dataset.filter(lambda x: x['language'] == 'python', num_proc=5))
        assert len(unbiased_dataset.filter(lambda x: x['language'] == 'java', num_proc=5)) == len(
            biased_dataset.filter(lambda x: x['language'] == 'java', num_proc=5))

        print(biased_dataset)
        print(unbiased_dataset)

        new_dataset = DatasetDict({"biased": biased_dataset,
                                   "unbiased": unbiased_dataset})
        print(new_dataset)
        new_dataset.push_to_hub(args.hf_dataset, private=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='codesearchnet')
    parser.add_argument('--first_stage', action='store_true')
    parser.add_argument('--hf_dataset', type=str, required=False)
    parser.add_argument('--samples', type=int, default=800_000)
    args = parser.parse_args()
    main(args)
