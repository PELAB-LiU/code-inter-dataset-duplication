import sys

sys.path.append("..")

from codesearchnet.load_dataset import load_datasets

import argparse
import json
import sqlite3

import yaml
from tqdm import tqdm

from bigclonebench.utils import filter_tokens

LANG = 'java'

with open('metadata.yaml', 'r') as file:
    DATASET_METADATA = yaml.safe_load(file)


def register_database(database, multimodal, unimodal):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # define the data to be inserted as a tuple
    tuple_dataset = (DATASET_METADATA['id'],
                     DATASET_METADATA['url'],
                     DATASET_METADATA['tasks'])

    # execute the insert statement
    cursor.execute("INSERT INTO datasets (id, url, tasks) VALUES (?, ?, ?)", tuple_dataset)

    for _, data in tqdm(multimodal.iterrows()):
        f_tokens = filter_tokens(data['code_tokens'], LANG)
        tuple_snippet = (data['code'], DATASET_METADATA['id'], LANG, json.dumps(list(data['code_tokens'])),
                         json.dumps(list(f_tokens)), data['docstring'])
        cursor.execute("INSERT INTO snippets (snippet, dataset, "
                       "language, tokens, filtered_tokens, nl) VALUES (?, ?, ?, ?, ?, ?)",
                       tuple_snippet)

    for _, data in tqdm(unimodal.iterrows()):
        f_tokens = filter_tokens(data['function_tokens'], LANG)
        tuple_snippet = (data['function'], DATASET_METADATA['id'], LANG,
                         json.dumps(list(data['function_tokens'])),
                         json.dumps(list(f_tokens)))
        cursor.execute("INSERT INTO snippets (snippet, dataset, "
                       "language, tokens, filtered_tokens) VALUES (?, ?, ?, ?, ?)",
                       tuple_snippet)
    conn.commit()
    conn.close()


def main(args):
    # dataset = load_dataset('CM/codexglue_code2text_java')
    # dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']]).to_pandas()
    multimodal, unimodal = load_datasets(lang="java")
    register_database(args.db, multimodal, unimodal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='../interduplication.db')
    args = parser.parse_args()
    main(args)
