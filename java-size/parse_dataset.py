import glob
import json
import sys

from tqdm import tqdm

sys.path.append('..')

from utils import ParseLog, get_methods_java, get_tokens_from_snippet

contents = []
log = ParseLog()
for f in tqdm(glob.glob('java-med/**/*.java', recursive=True)):
    try:
        with open(f, 'r') as file:
            content = file.read()
    except:
        log.register_fail_file()
        continue
    contents.append(content)


all_data = []
i = 0
for content in tqdm(contents):
    try:
        methods = get_methods_java(content)
        log.register_success_file()
    except:
        print(f'Failed to parse snippet')
        log.register_fail_file()
        continue
    for body in methods:
        try:
            all_data.append({"id_within_dataset": i,
                             "snippet": body,
                             "tokens": get_tokens_from_snippet(body, 'java')})
            i += 1
            log.register_success_snippet()
        except:
            print(f'Failed to parse a method of snippet')
            log.register_fail_snippet()
            continue
    if i % 1000 == 0 and i > 0:
        print(f'Parsed {i} methods')

with open('data.jsonl', 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')

log.save_log('log.json')

