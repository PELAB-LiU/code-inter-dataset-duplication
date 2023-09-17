import json
import sqlite3

import networkx as nx
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATA_PATH = 'data.jsonl'
DATASET = 'bigclonebench'
DATASET_CSN = 'codesearchnet'
DB = '../interduplication.db'
INTER = 'interduplicates.json'
REPR = 'representatives.json'
TRAIN_PATH, TEST_PATH, VALID_PATH = 'train.txt', 'test.txt', 'valid.txt'
TRAIN_JSONL_PATH, TEST_JSONL_PATH, VALID_JSONL_PATH = 'train.jsonl', 'test.jsonl', 'valid.jsonl'
HF_PATH = 'antolin/bigclonebench_interduplication'
UPLOAD = True
DB_ACCESS = False

with open(DATA_PATH, 'r') as json_file:
    json_list = list(json_file)
data = [json.loads(json_str) for json_str in json_list]
data = {d['id_within_dataset']: d for d in data}


def read_pairs(path):
    # read pairs format: p1\tp2\tlabel
    with open(path, 'r') as file:
        lines = file.readlines()
    pairs = [line.strip().split('\t') for line in lines]
    pairs = [(int(pair[0]), int(pair[1]), int(pair[2])) for pair in pairs]
    return pairs


def load_graph():
    conn = sqlite3.connect(DB)
    query = f'SELECT id, dataset, id_within_dataset, split_within_dataset ' \
            f'FROM snippets WHERE dataset = "{DATASET}" or dataset = "{DATASET_CSN}"'
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


train_pairs = read_pairs(TRAIN_PATH)
train_pairs = [(x1, x2, l) for x1, x2, l in train_pairs if x1 in data and x2 in data]
test_pairs = read_pairs(TEST_PATH)
test_pairs = [(x1, x2, l) for x1, x2, l in test_pairs if x1 in data and x2 in data]
valid_pairs = read_pairs(VALID_PATH)
valid_pairs = [(x1, x2, l) for x1, x2, l in valid_pairs if x1 in data and x2 in data]

interduplicates = []

if DB_ACCESS:
    G = load_graph()
    for n1, n2 in G.edges(data=False):
        if G.nodes[n1]['dataset'] == DATASET_CSN and G.nodes[n2]['dataset'] == DATASET:
            interduplicates.append(G.nodes[n2]['id_within_dataset'])
        if G.nodes[n1]['dataset'] == DATASET and G.nodes[n2]['dataset'] == DATASET_CSN:
            interduplicates.append(G.nodes[n1]['id_within_dataset'])
else:
    with open(INTER, 'r') as json_file:
        interduplicates = json.load(json_file)
    with open(REPR, 'r') as json_file:
        representatives = set(json.load(json_file))
    train_pairs = [(x1, x2, l) for x1, x2, l in train_pairs if x1 in representatives and x2 in representatives]
    test_pairs = [(x1, x2, l) for x1, x2, l in test_pairs if x1 in representatives and x2 in representatives]
    valid_pairs = [(x1, x2, l) for x1, x2, l in valid_pairs if x1 in representatives and x2 in representatives]

interduplicates = set(interduplicates)

print('Length of train pairs: ', len(train_pairs))
print('Length of train positive pairs: ', len([pair for pair in train_pairs if pair[2] == 1]))
print('Length of test pairs: ', len(test_pairs))
print('Length of test positive pairs: ', len([pair for pair in test_pairs if pair[2] == 1]))
print('Length of valid pairs: ', len(valid_pairs))
print('Length of valid positive pairs: ', len([pair for pair in valid_pairs if pair[2] == 1]))

problematic_pairs_train = [pair for pair in train_pairs if pair[0] in interduplicates or pair[1] in interduplicates]
problematic_pairs_test = [pair for pair in test_pairs if pair[0] in interduplicates or pair[1] in interduplicates]
problematic_pairs_valid = [pair for pair in valid_pairs if pair[0] in interduplicates or pair[1] in interduplicates]

print('Length of problematic train pairs: ', len(problematic_pairs_train))
print('Length of problematic train positive pairs: ', len([pair for pair in problematic_pairs_train if pair[2] == 1]))
print('Length of problematic test pairs: ', len(problematic_pairs_test))
print('Length of problematic test positive pairs: ', len([pair for pair in problematic_pairs_test if pair[2] == 1]))
print('Length of problematic valid pairs: ', len(problematic_pairs_valid))
print('Length of problematic valid positive pairs: ', len([pair for pair in problematic_pairs_valid if pair[2] == 1]))

print(f'Percentage of problematic train pairs: {len(problematic_pairs_train) * 100 / len(train_pairs):.2f}')
print(f'Percentage of problematic test pairs: {len(problematic_pairs_test) * 100 / len(test_pairs):.2f}')
print(f'Percentage of problematic valid pairs: {len(problematic_pairs_valid) * 100 / len(valid_pairs):.2f}')

very_problematic_pairs_train = [pair for pair in train_pairs if pair[0] in interduplicates and pair[1] in interduplicates]
very_problematic_pairs_test = [pair for pair in test_pairs if pair[0] in interduplicates and pair[1] in interduplicates]
very_problematic_pairs_valid = [pair for pair in valid_pairs if pair[0] in interduplicates and pair[1] in interduplicates]

print('Length of very problematic train pairs: ', len(very_problematic_pairs_train))
print('Length of very problematic train positive pairs: ', len([pair for pair in very_problematic_pairs_train if pair[2] == 1]))
print('Length of very problematic test pairs: ', len(very_problematic_pairs_test))
print('Length of very problematic test positive pairs: ', len([pair for pair in very_problematic_pairs_test if pair[2] == 1]))
print('Length of very problematic valid pairs: ', len(very_problematic_pairs_valid))
print('Length of very problematic valid positive pairs: ', len([pair for pair in very_problematic_pairs_valid if pair[2] == 1]))

if UPLOAD:
    train_examples = []
    for pair in train_pairs:
        tokens1 = data[pair[0]]['tokens']
        tokens2 = data[pair[1]]['tokens']
        label = pair[2]
        inteduplicate = pair[0] in interduplicates or pair[1] in interduplicates
        example = {'tokens1': tokens1, 'tokens2': tokens2, 'label': label, 'is_duplicated': inteduplicate}
        train_examples.append(example)

    test_examples = []
    for pair in test_pairs:
        tokens1 = data[pair[0]]['tokens']
        tokens2 = data[pair[1]]['tokens']
        label = pair[2]
        inteduplicate = pair[0] in interduplicates or pair[1] in interduplicates
        example = {'tokens1': tokens1, 'tokens2': tokens2, 'label': label, 'is_duplicated': inteduplicate}
        test_examples.append(example)

    valid_examples = []
    for pair in valid_pairs:
        tokens1 = data[pair[0]]['tokens']
        tokens2 = data[pair[1]]['tokens']
        label = pair[2]
        inteduplicate = pair[0] in interduplicates or pair[1] in interduplicates
        example = {'tokens1': tokens1, 'tokens2': tokens2, 'label': label, 'is_duplicated': inteduplicate}
        valid_examples.append(example)

    with open(TRAIN_JSONL_PATH, 'w') as outfile:
        for example in train_examples:
            json.dump(example, outfile)
            outfile.write('\n')

    with open(TEST_JSONL_PATH, 'w') as outfile:
        for example in test_examples:
            json.dump(example, outfile)
            outfile.write('\n')

    with open(VALID_JSONL_PATH, 'w') as outfile:
        for example in valid_examples:
            json.dump(example, outfile)
            outfile.write('\n')

    dataset = load_dataset('json', data_files={"train": TRAIN_JSONL_PATH,
                                               "test": TEST_JSONL_PATH,
                                               "valid": VALID_JSONL_PATH})
    print(dataset)
    dataset.push_to_hub(HF_PATH, private=True)
