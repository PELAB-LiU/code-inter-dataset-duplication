import json
import sys

from tqdm import tqdm

sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog

FILE = 'cxg_data.jsonl'
OUT = 'data.jsonl'


def load_dataset(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    result = [json.loads(json_str) for json_str in json_list]
    return result


dataset = load_dataset(FILE)
new_dataset = []
log = ParseLog()
for data in tqdm(dataset):
    try:
        tokens = get_tokens_from_snippet(data['func'], 'java')
        log.register_success_snippet()
    except:
        print(f'Error with {data["idx"]}')
        log.register_fail_snippet()
        continue
    new_dataset.append({'snippet': data['func'],
                        'tokens': tokens,
                        'id_within_dataset': int(data['idx'])})

with open(OUT, "w") as outfile:
    for item in new_dataset:
        json.dump(item, outfile)
        outfile.write("\n")

log.save_log('log.json')
