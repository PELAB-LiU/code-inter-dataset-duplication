import argparse
import json
import os.path
import sqlite3

import networkx as nx
import pandas as pd
from tqdm import tqdm

from dataset_overlapping import load_graph


def main(args):
    G = load_graph(args.db, args.lang)
    target_dataset = args.target_dataset

    dup_ids = set([])
    for n1, n2 in tqdm(G.edges, desc='Computing overlapping'):
        d1 = G.nodes[n1]['dataset']
        d2 = G.nodes[n2]['dataset']
        if d1 != d2:
            if d1 == target_dataset:
                dup_ids.add(n1)
            if d2 == target_dataset:
                dup_ids.add(n2)

    dup_ids_within_dataset = [G.nodes[n]['id_within_dataset'] for n in dup_ids]
    print(f'Percentage {100 * len(dup_ids_within_dataset)/len([n for n in G if G.nodes[n]["dataset"] == target_dataset]):.2f}')
    with open(os.path.join(target_dataset, f'dups_{args.lang}.json'), 'w') as f:
        json.dump(dup_ids_within_dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='interduplication.db')
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--target_dataset', type=str, default='codesearchnet')
    args = parser.parse_args()
    main(args)
