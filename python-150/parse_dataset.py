import json
import sys

DESCRIPTIONS_FILE = 'descriptions.txt'
SNIPPETS_FILE = 'snippets.txt'
OUTPUT = 'data.jsonl'

# setting path
sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog

all_data = []
i = 0
log = ParseLog()
with open(DESCRIPTIONS_FILE, 'r') as file1, open(SNIPPETS_FILE, 'r') as file2:
    for description, code in zip(file1, file2):
        try:
            all_data.append({"id_within_dataset": i,
                             "snippet": code,
                             "tokens": get_tokens_from_snippet(code, 'python'),
                             "nl": description})
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
