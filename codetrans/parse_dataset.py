import json
import sys

FILE_JAVA = 'all.java'
FILE_CS = 'all.cs'
OUTPUT = 'data.jsonl'

# setting path
sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog

all_data = []
i = 0
log = ParseLog()
with open(FILE_JAVA, 'r') as file1, open(FILE_CS, 'r') as file2:
    for java, cs in zip(file1, file2):
        try:
            all_data.append({"id_within_dataset": i,
                             "snippet": java,
                             "tokens": get_tokens_from_snippet(java, 'java'),
                             "cs": cs})
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
