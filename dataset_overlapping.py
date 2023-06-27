import argparse
import sqlite3

import networkx as nx
import pandas as pd
from tqdm import tqdm


def load_graph(db, lang):
    conn = sqlite3.connect(db)
    query = f'SELECT id, dataset FROM snippets WHERE language = "{lang}"'
    df = pd.read_sql_query(query, conn)
    vertices = df.to_records(index=False).tolist()
    query = f'SELECT snippet1, snippet2 FROM duplicates'
    df = pd.read_sql_query(query, conn)
    edges = df.to_records(index=False).tolist()

    G = nx.Graph()
    for x, d in tqdm(vertices, desc='Adding nodes'):
        G.add_node(x, dataset=d)
    # add the edges to the graph
    for edge in tqdm(edges, desc='Adding edges'):
        G.add_edge(*edge)
    return G


def ovelapping(G, source_dataset, target_dataset):
    result = []
    relevant_nodes = [n for n, d in G.nodes(data=True) if d['dataset'] == source_dataset
                      or d['dataset'] == target_dataset]
    G_filtered = G.subgraph(relevant_nodes)
    for n1, n2 in tqdm(G_filtered.edges, desc='Computing overlapping'):
        d1 = G_filtered.nodes[n1]['dataset']
        d2 = G_filtered.nodes[n2]['dataset']
        if d1 == source_dataset and d2 == target_dataset:
            result.append(n1)
        elif d1 == target_dataset and d2 == source_dataset:
            result.append(n2)
    return len(set(result)) / len([n for n in G_filtered.nodes if G_filtered.nodes[n]['dataset'] == source_dataset])


def main(args):
    G = load_graph(args.db, args.lang)
    datasets = set([d for n, d in G.nodes(data=True)])
    for source_dataset in tqdm(datasets, desc='Computing overlapping'):
        for target_dataset in datasets:
            if source_dataset != target_dataset:
                print(f'{source_dataset} - {target_dataset}: {ovelapping(G, source_dataset, target_dataset)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='interduplication.db')
    parser.add_argument('--lang', type=str, default='java')
    args = parser.parse_args()
    main(args)
