import argparse
import json
import os.path
import sqlite3

import networkx as nx
import pandas as pd
from tqdm import tqdm


def load_graph(db, lang):
    conn = sqlite3.connect(db)
    query = f'SELECT id, dataset, id_within_dataset, split_within_dataset FROM snippets WHERE language = "{lang}"'
    df = pd.read_sql_query(query, conn)
    vertices = df.to_records(index=False).tolist()
    query = f'SELECT snippet1, snippet2 FROM duplicates'
    df = pd.read_sql_query(query, conn)
    edges = df.to_records(index=False).tolist()

    G = nx.Graph()
    for x, d, id_within_dataset, split_within_dataset in tqdm(vertices, desc='Adding nodes'):
        G.add_node(x, dataset=d, id_within_dataset=id_within_dataset, split_within_dataset=split_within_dataset)
    # add the edges to the graph
    for edge in tqdm(edges, desc='Adding edges'):
        if edge[0] not in G.nodes():
            continue
        if edge[1] not in G.nodes():
            continue
        G.add_edge(*edge)
    return G


def overlapping(G, source_dataset, target_dataset):
    result = []
    ids_within_dataset = []
    relevant_nodes = [n for n, d in G.nodes(data=True) if d['dataset'] == source_dataset
                      or d['dataset'] == target_dataset]
    G_filtered = G.subgraph(relevant_nodes)
    for n1, n2 in tqdm(G_filtered.edges, desc='Computing overlapping'):
        d1 = G_filtered.nodes[n1]['dataset']
        d2 = G_filtered.nodes[n2]['dataset']
        if d1 == source_dataset and d2 == target_dataset:
            result.append(n1)
            ids_within_dataset.append(G_filtered.nodes[n1]['id_within_dataset'])
        elif d1 == target_dataset and d2 == source_dataset:
            result.append(n2)
            ids_within_dataset.append(G_filtered.nodes[n2]['id_within_dataset'])

    # REPORT
    print(f'Source dataset: {source_dataset}, Target dataset: {target_dataset}')

    number_duplicates = len(set(result))
    size_dataset = len([n for n in G_filtered.nodes if G_filtered.nodes[n]['dataset'] == source_dataset])
    print(f'Number of duplicates: {number_duplicates}')

    full_percentage = number_duplicates / size_dataset
    print(f'Percentage of duplication within source dataset: {full_percentage * 100:.2f}%')

    duplicates_test_set = [n for n in set(result) if G_filtered.nodes[n]['split_within_dataset'] == 'test']
    test_set = [n for n in G_filtered.nodes if G_filtered.nodes[n]['split_within_dataset'] == 'test'
                and G_filtered.nodes[n]['dataset'] == source_dataset]
    if len(test_set) == 0:
        print(f'Test set not defined in {source_dataset}')
    else:
        print(f'Number of duplicates in test set: {len(duplicates_test_set)}')
        print(f'Percentage of duplication within test set: {len(duplicates_test_set) / len(test_set) * 100:.2f}%')
    return list(set(ids_within_dataset))


def get_representative(G):
    datasets = set([d['dataset'] for n, d in G.nodes(data=True)])
    representatives = []
    for dataset in datasets:
        relevant_nodes = [n for n, d in G.nodes(data=True) if d['dataset'] == dataset]
        G_filtered = G.subgraph(relevant_nodes)
        for c in nx.connected_components(G_filtered):
            c = list(c)
            representatives.append(c[0])
    return G.subgraph(representatives)


def main(args):
    G = load_graph(args.db, args.lang)
    G = get_representative(G)
    datasets = set([d['dataset'] for n, d in G.nodes(data=True) if d['dataset'] != args.target_dataset])
    target_dataset = args.target_dataset
    for source_dataset in tqdm(datasets, desc='Computing overlapping'):
        nodes = overlapping(G, source_dataset, target_dataset)
        with open(os.path.join(source_dataset, 'interduplicates.json'), 'w') as f:
            json.dump(nodes, f)
        with open(os.path.join(source_dataset, 'representatives.json'), 'w') as f:
            json.dump([d['id_within_dataset'] for n, d in G.nodes(data=True)
                       if d['dataset'] == source_dataset], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='interduplication.db')
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--target_dataset', type=str, default='codesearchnet')
    args = parser.parse_args()
    main(args)
