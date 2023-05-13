import argparse
import json
import sqlite3
from itertools import combinations

import pandas as pd
from dpu_utils.codeutils.deduplication import DuplicateDetector
from tqdm import tqdm

LANGUAGES = ['java', 'python']


def register_database(duplicates, conn):
    cursor = conn.cursor()
    for group in duplicates:
        for i, j in combinations(group, 2):
            cursor.execute("INSERT INTO duplicates (snippet1, snippet2) VALUES (?, ?)", (i, j))


def main(args):
    conn = sqlite3.connect(args.db)

    print(f'Compute duplicates of {args.lang}')
    detector = DuplicateDetector(min_num_tokens_per_document=5)
    query = f'SELECT id, tokens FROM snippets WHERE language = "{args.lang}"'
    df = pd.read_sql_query(query, conn)
    for _, row in tqdm(df.iterrows(), desc='Add files'):
        detector.add_file(row['id'], json.loads(row['tokens']), language=args.lang)
    duplicates = detector.compute_duplicates()
    detector.print_clone_set_stats(duplicates)
    num_cloned_files = sum(len(c) for c in duplicates)
    print('Number of cloned files:', num_cloned_files)
    register_database(duplicates, conn)

    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='./interduplication.db')
    parser.add_argument('--lang', type=str, required=True, choices=LANGUAGES)
    args = parser.parse_args()
    main(args)
