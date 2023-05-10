import json

import javalang
from tqdm import tqdm

FILE = 'data.jsonl'


def load_dataset(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    result = [json.loads(json_str) for json_str in json_list]
    return result


dataset = load_dataset(FILE)
new_dataset = []

for data in tqdm(dataset):
    tokens = javalang.tokenizer.tokenize(data['func'])
    tokens = [str(t.value) for t in tokens]
    new_dataset.append({'snippet': data['func'],
                        'tokens': tokens,
                        'id_within_dataset': int(data['idx'])})

with open(FILE, "w") as outfile:
    for item in new_dataset:
        json.dump(item, outfile)
        outfile.write("\n")
