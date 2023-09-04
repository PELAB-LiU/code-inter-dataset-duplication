import argparse
import json
import sqlite3

import yaml
from tqdm import tqdm


def load_metadata(file):
    with open(file, 'r') as file:
        metadata = yaml.safe_load(file)
    return metadata


def load_dataset(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    result = [json.loads(json_str) for json_str in json_list]
    return result


def register_database(database, dataset, metadata):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # define the data to be inserted as a tuple
    tuple_dataset = (metadata['id'],
                     metadata['url'])

    # execute the insert statement
    cursor.execute("INSERT INTO datasets (id, url) VALUES (?, ?)", tuple_dataset)

    for data in tqdm(dataset):
        tuple_snippet = (data['snippet'],
                         metadata['id'],
                         data['language'] if 'language' in data else metadata['language'],
                         data['id_within_dataset'],
                         json.dumps(data['tokens']),
                         data['split_within_dataset'] if 'split_within_dataset' in data else None)
        cursor.execute("INSERT INTO snippets (snippet, dataset, "
                       "language, id_within_dataset, tokens, split_within_dataset) VALUES (?, ?, ?, ?, ?, ?)",
                       tuple_snippet)
    conn.commit()
    conn.close()


def main(args):
    dataset = load_dataset(args.data)
    metadata = load_metadata(args.meta)
    register_database(args.db, dataset, metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='interduplication.db')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--meta', type=str, required=True)
    args = parser.parse_args()
    main(args)
