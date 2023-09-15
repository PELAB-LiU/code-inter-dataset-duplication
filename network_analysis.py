import argparse

from dataset_overlapping import load_graph


def analysis(G, small_dataset, big_dataset):
    test_nodes = set([n for n in G if G.nodes[n]['dataset'] == small_dataset
                      and G.nodes[n]['split_within_dataset'] == 'test'])
    train_test_dup = []
    pretrain_test_dup = []
    for u, v in G.edges():
        if G.nodes[u]['dataset'] == small_dataset and G.nodes[v]['dataset'] == small_dataset:
            if G.nodes[u]['split_within_dataset'] == 'test' and G.nodes[v]['split_within_dataset'] == 'train':
                train_test_dup.append(u)
            elif G.nodes[v]['split_within_dataset'] == 'test' and G.nodes[u]['split_within_dataset'] == 'train':
                train_test_dup.append(v)
        if G.nodes[u]['dataset'] == big_dataset and G.nodes[v]['dataset'] == small_dataset \
                and G.nodes[v]['split_within_dataset'] == 'test':
            pretrain_test_dup.append(v)
        elif G.nodes[v]['dataset'] == big_dataset and G.nodes[u]['dataset'] == small_dataset \
                and G.nodes[u]['split_within_dataset'] == 'test':
            pretrain_test_dup.append(u)

    train_test_dup = set(train_test_dup)
    pretrain_test_dup = set(pretrain_test_dup)

    print(f'Proportion train-test dup within test: {len(train_test_dup) / len(test_nodes)}')
    print(f'Proportion pretrain-test dup within test: {len(pretrain_test_dup) / len(test_nodes)}')
    print(f'Proportion pretrain-test dup within train-test dup: '
          f'{len(pretrain_test_dup.intersection(train_test_dup)) / len(train_test_dup)}')
    print(f'Proportion train-test dup within pretrain-test dup: '
          f'{len(pretrain_test_dup.intersection(train_test_dup)) / len(pretrain_test_dup)}')


def main(args):
    G = load_graph(args.db, args.lang)
    datasets = [args.big_dataset, args.small_dataset]
    G_filtered = G.subgraph([n for n in G if G.nodes[n]['dataset'] in datasets])
    print(f'Nodes: {len(G_filtered)}')
    print(f'Edges: {len(G_filtered.edges)}')
    analysis(G_filtered, args.small_dataset, args.big_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='interduplication.db')
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--big_dataset', type=str, default='codesearchnet')
    parser.add_argument('--small_dataset', type=str, default='tlc')
    args = parser.parse_args()
    main(args)
