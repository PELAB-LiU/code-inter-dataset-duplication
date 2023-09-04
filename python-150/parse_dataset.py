import json
import re
import sys

CODE_FILES = 'data_ps.declbodies.'
DESC_FILES = 'data_ps.descriptions.'
SPLITS = ['train', 'valid', 'test']
NEW_LINE_TOKEN = 'DCNL'
SPACE_TOKEN = 'DCSP'
OUTPUT = 'data.jsonl'

# setting path
sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog

all_data = []
i = 0
log = ParseLog()
for split in SPLITS:
    DESCRIPTIONS_FILE = DESC_FILES + split
    SNIPPETS_FILE = CODE_FILES + split
    print('Processing ' + DESCRIPTIONS_FILE + ' and ' + SNIPPETS_FILE)
    with open(DESCRIPTIONS_FILE, 'r', errors='ignore') as file1, open(SNIPPETS_FILE, 'r', errors='ignore') as file2:
        for description, code in zip(file1, file2):
            description = re.sub(' +', ' ', description).replace(SPACE_TOKEN, ' ').replace(NEW_LINE_TOKEN, '\n')[1:-2]
            code = re.sub(' +', ' ', code).replace(SPACE_TOKEN, ' ').replace(NEW_LINE_TOKEN, '\n')
            try:
                all_data.append({"id_within_dataset": i,
                                 "snippet": code,
                                 "tokens": get_tokens_from_snippet(code, 'python'),
                                 "nl": description,
                                 "split_within_dataset": split})
                i += 1
                log.register_success_snippet()
            except:
                print(f'Failed to parse a snippet. Skipping...')
                log.register_fail_snippet()
                continue

with open(OUTPUT, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')

log.save_log('log.json')
