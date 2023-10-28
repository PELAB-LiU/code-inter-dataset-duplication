import argparse
import string
from collections import Counter

from tqdm import tqdm

from dataset_overlapping import load_graph
from register_in_db import load_dataset

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--db', default='interduplication.db')
args = parser.parse_args()

G = load_graph(args.db, args.lang)
# filter graph
G = G.subgraph([n for n, d in G.nodes(data=True) if d['dataset'] == args.dataset_name or d['dataset'] == 'codesearchnet'])

target_dataset = load_dataset(args.data)
codesearchnet_dataset = load_dataset('codesearchnet/data.jsonl')


def remove_punctuation(input_string):
    # Create a translation table to remove punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # Use translate() to remove punctuation
    cleaned_string = input_string.translate(translator)

    return cleaned_string


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def jaccard_similarity_included(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s2))


def jaccard_similarity_bags(bag1, bag2):
    # Convert the bags to Counters, which count the occurrences of elements
    counter1 = Counter(bag1)
    counter2 = Counter(bag2)

    # Calculate the intersection by taking the element-wise minimum of Counters
    intersection = sum((counter1 & counter2).values())

    # Calculate the union by taking the element-wise maximum of Counters
    union = sum((counter1 | counter2).values())

    # Calculate Jaccard similarity
    if union == 0:
        return 0  # To handle the case where both bags are empty
    else:
        return intersection / union


# enrich graph with nl descriptions
for n, d in tqdm(G.nodes(data=True), desc='Enriching graph'):
    if d['dataset'] == args.dataset_name:
        d['nl'] = target_dataset[d['id_within_dataset']]['nl']
    elif d['dataset'] == 'codesearchnet':
        d['nl'] = codesearchnet_dataset[d['id_within_dataset']]['nl']

# compare nl descriptions of duplicates
jaccard_similarities = []
jaccard_similarities_bag = []
jaccard_similarities_included = []
lens_csn = []
lens_target_dataset = []
empty_strings = 0
edges = 0
for n1, n2 in tqdm(G.edges, desc='Computing Jaccard similarity'):
    d1 = G.nodes[n1]['dataset']
    d2 = G.nodes[n2]['dataset']
    if d1 == args.dataset_name and d2 == 'codesearchnet':
        nl_target = G.nodes[n1]['nl']
        nl_csn = G.nodes[n2]['nl']
    elif d1 == 'codesearchnet' and d2 == args.dataset_name:
        nl_target = G.nodes[n2]['nl']
        nl_csn = G.nodes[n1]['nl']
    else:
        continue
    if nl_csn == '':
        empty_strings += 1
    else:
        csn_cleaned = remove_punctuation(nl_csn).lower().split()
        target_cleaned = remove_punctuation(nl_target).lower().split()
        jaccard_similarities.append(jaccard_similarity(csn_cleaned, target_cleaned))
        jaccard_similarities_bag.append(jaccard_similarity_bags(csn_cleaned, target_cleaned))
        jaccard_similarities_included.append(jaccard_similarity_included(csn_cleaned, target_cleaned))
        lens_csn.append(len(csn_cleaned))
        lens_target_dataset.append(len(target_cleaned))
    edges += 1

print(f'Average Jaccard similarity: {sum(jaccard_similarities) / len(jaccard_similarities):.4f}')
print(f'Average Jaccard similarity (includegd): {sum(jaccard_similarities_included) / len(jaccard_similarities_included):.4f}')
print(f'Average Jaccard similarity (bag): {sum(jaccard_similarities_bag) / len(jaccard_similarities_bag):.4f}')
print(f'Percentage of empty strings: {empty_strings / edges * 100:.2f}%')
print(f'Average length of nl descriptions (csn): {sum(lens_csn) / len(lens_csn):.4f}')
print(f'Average length of nl descriptions (target): {sum(lens_target_dataset) / len(lens_target_dataset):.4f}')
