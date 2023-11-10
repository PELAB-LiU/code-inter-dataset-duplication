import json
import sys

from tqdm import tqdm

sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog

SPLITS = ['train', 'valid', 'test']
OUTPUT = 'data.jsonl'

data = []
for split in SPLITS:
    code_file = f'{split}.json'
    with open(code_file, 'r') as json_file:
        json_list = list(json_file)
    result = [json.loads(json_str) for json_str in json_list]

    nl_file = f'{split}.token.nl'
    with open(nl_file, 'r') as csv_file:
        csv_list = list(csv_file)
    nl_list = [csv_str.split('\t')[-1] for csv_str in csv_list]

    for j, r in enumerate(result):
        r['split_within_dataset'] = split
        r['nl'] = nl_list[j]
    data += result

all_data = []
i = 0
log = ParseLog()
for sample in tqdm(data, desc=f'Parsing code snippets'):
    code = sample['code']
    nl = sample['nl']
    split_within_dataset = sample['split_within_dataset']
    try:
        all_data.append({"id_within_dataset": i,
                         "snippet": code,
                         "tokens": get_tokens_from_snippet(code, 'java'),
                         "nl": nl,
                         "split_within_dataset": split_within_dataset})
        i += 1
        log.register_success_snippet()
    except:
        log.register_fail_snippet()
        continue

log.save_log('log.json')

with open(OUTPUT, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')
