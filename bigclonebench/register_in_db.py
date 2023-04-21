import argparse
import json
import sqlite3

import javalang
import yaml
from tqdm import tqdm

from utils import filter_tokens

DATA = 'data.jsonl'
LANG = 'java'

with open('metadata.yaml', 'r') as file:
    DATASET_METADATA = yaml.safe_load(file)


def load_dataset(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    result = [json.loads(json_str) for json_str in json_list]
    return result


def register_database(database, dataset):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # define the data to be inserted as a tuple
    tuple_dataset = (DATASET_METADATA['id'],
                     DATASET_METADATA['url'],
                     DATASET_METADATA['tasks'])

    # execute the insert statement
    cursor.execute("INSERT INTO datasets (id, url, tasks) VALUES (?, ?, ?)", tuple_dataset)

    for data in tqdm(dataset):
        tokens = javalang.tokenizer.tokenize(data['func'])
        tokens = [str(t.value) for t in tokens]
        f_tokens = filter_tokens(tokens, LANG)
        tokens = json.dumps(tokens)
        f_tokens = json.dumps(f_tokens)
        tuple_snippet = (data['func'], DATASET_METADATA['id'], LANG, data['idx'], tokens, f_tokens)
        cursor.execute("INSERT INTO snippets (snippet, dataset, "
                       "language, id_within_dataset, tokens, filtered_tokens) VALUES (?, ?, ?, ?, ?, ?)",
                       tuple_snippet)
    conn.commit()
    conn.close()


def main(args):
    dataset = load_dataset(DATA)
    register_database(args.db, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='../interduplication.db')
    args = parser.parse_args()
    main(args)
