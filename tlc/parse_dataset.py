import json
import pickle
import sys

from tqdm import tqdm

sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog

INPUT_FILE = 'data.pkl'
OUTPUT = 'data.jsonl'

# load pickle file
with open(INPUT_FILE, 'rb') as file:
    data = pickle.load(file)

all_data = []
i = 0
log = ParseLog()
for key in data:
    for sample in tqdm(data[key], desc=f'Parsing {key} split'):
        sample = data[key][sample]
        code = sample['code']
        nl = ' '.join(sample['summary'])
        try:
            all_data.append({"id_within_dataset": i,
                             "snippet": code,
                             "tokens": get_tokens_from_snippet(code, 'java'),
                             "nl": nl})
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