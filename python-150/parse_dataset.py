import json
import re
import sys

SPLITS = ['train', 'valid', 'test']
OUTPUT = 'data.jsonl'

# setting path
sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog

all_data = []
i = 0
log = ParseLog()
for split in SPLITS:
    DESCRIPTIONS_FILE = f'attributes/{split}.docstring_tokens'
    SNIPPETS_FILE = f'attributes/{split}.code'
    print('Processing ' + DESCRIPTIONS_FILE + ' and ' + SNIPPETS_FILE)
    with open(DESCRIPTIONS_FILE, 'r', errors='ignore') as file1, open(SNIPPETS_FILE, 'r', errors='ignore') as file2:
        for description, code in zip(file1, file2):
            description = json.loads(description)
            code = json.loads(code).replace("\t", " ")
            all_data.append({"id_within_dataset": i,
                             "snippet": code,
                             "tokens": get_tokens_from_snippet(code, 'python'),
                             "nl": ' '.join(description),
                             "split_within_dataset": split})
            i += 1
            log.register_success_snippet()

with open(OUTPUT, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')

log.save_log('log.json')
