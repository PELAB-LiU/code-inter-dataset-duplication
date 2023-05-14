import json
import sys

DESCRIPTIONS_FILE = 'descriptions.txt'
SNIPPETS_FILE = 'snippets.txt'
OUTPUT = 'data.jsonl'

# setting path
sys.path.append('..')
from utils import get_tokens_from_snippet


all_data = []
i = 0
with open(DESCRIPTIONS_FILE, 'r') as file1, open(SNIPPETS_FILE, 'r') as file2:
    for description, code in zip(file1, file2):
        all_data.append({"id_within_dataset": i,
                         "snippet": code,
                         "tokens": get_tokens_from_snippet(code, 'python'),
                         "nl": description})
        i += 1

with open(OUTPUT, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')
