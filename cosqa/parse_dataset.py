import json

import javalang
from tqdm import tqdm

FILE = 'data.jsonl'


def load_dataset(path):
    with open(path) as f:
        result = json.load(f)
    return result


dataset = load_dataset('cosqa-all.json')
new_dataset = []

for i, data in tqdm(enumerate(dataset)):
    new_dataset.append({'snippet': data['code'],
                        'tokens': data['code_tokens'].split(' '),
                        'id_within_dataset': i,
                        "nl": data['doc']})

with open(FILE, "w") as outfile:
    for item in new_dataset:
        json.dump(item, outfile)
        outfile.write("\n")
