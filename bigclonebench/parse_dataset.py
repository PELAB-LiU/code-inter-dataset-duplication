import json
import sys

from tqdm import tqdm

sys.path.append('..')
from utils import get_tokens_from_snippet

FILE = 'data.jsonl'


def load_dataset(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    result = [json.loads(json_str) for json_str in json_list]
    return result


dataset = load_dataset(FILE)
new_dataset = []

for data in tqdm(dataset):
    try:
        tokens = get_tokens_from_snippet(data['func'], 'java')
    except:
        print(f'Error with {data["idx"]}')
        continue
    new_dataset.append({'snippet': data['func'],
                        'tokens': tokens,
                        'id_within_dataset': int(data['idx'])})

with open(FILE, "w") as outfile:
    for item in new_dataset:
        json.dump(item, outfile)
        outfile.write("\n")
