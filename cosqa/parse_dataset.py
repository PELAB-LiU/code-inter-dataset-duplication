import json
import sys

from tqdm import tqdm
# setting path
sys.path.append('..')
from utils import get_tokens_from_snippet

FILE = 'data.jsonl'


def load_dataset(path):
    with open(path) as f:
        result = json.load(f)
    return result


dataset = load_dataset('cosqa-all.json')
new_dataset = []

for i, data in tqdm(enumerate(dataset)):
    try:
        new_dataset.append({'snippet': data['code'],
                        'tokens': get_tokens_from_snippet(data['code'], 'python'),
                        'id_within_dataset': i,
                        "nl": data['doc']})
    except:
        print(f'Failed to parse snippet {i}')
        continue

with open(FILE, "w") as outfile:
    for item in new_dataset:
        json.dump(item, outfile)
        outfile.write("\n")
