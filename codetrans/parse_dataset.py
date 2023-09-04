import json
import sys

FILES_JAVA = ['train.java-cs.txt.java', 'valid.java-cs.txt.java', 'test.java-cs.txt.java']
FILES_CS = ['train.java-cs.txt.cs', 'valid.java-cs.txt.cs', 'test.java-cs.txt.cs']
OUTPUT = 'data.jsonl'

# setting path
sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog

all_data = []
i = 0
log = ParseLog()

for FILE_JAVA, FILE_CS in zip(FILES_JAVA, FILES_CS):
    split = FILE_JAVA.split('.')[0]
    with open(FILE_JAVA, 'r') as file1, open(FILE_CS, 'r') as file2:
        for java, cs in zip(file1, file2):
            try:
                all_data.append({"id_within_dataset": i,
                                 "snippet": java,
                                 "tokens": get_tokens_from_snippet(java, 'java'),
                                 "cs": cs,
                                 "split_within_dataset": split})
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
