import ast
import json
import os.path
import sys
from lib2to3 import refactor

from tqdm import tqdm

sys.path.append('..')
from utils import get_tokens_from_snippet, get_methods_java, ParseLog


def add_new_lines(code):
    contents = code.split(' ')
    new_contents = []
    for token in contents:
        new_contents.append(token)
        if token == '}' or token == '{':
            new_contents.append('\n')
    return ' '.join(new_contents)


PREFIX = 'token_completion'
OUTPUT = 'data.jsonl'

file_contents = {}
with open(f'{PREFIX}/test.txt', 'r') as file:
    file_contents['text'] = file.readlines()
with open(f'{PREFIX}/valid.txt', 'r') as file:
    file_contents['valid'] = file.readlines()
with open(f'{PREFIX}/train.txt', 'r') as file:
    file_contents['train'] = file.readlines()

all_data = []
i = 0
log = ParseLog()
for split in file_contents:
    for content in tqdm(file_contents[split], desc=f'Parsing {split} split'):
        content = ' '.join(content.split(' ')[1:-1])
        content = add_new_lines(content)
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
                                 "tokens": get_tokens_from_snippet(body, 'java'),
                                 "split_within_dataset": split})
                i += 1
                log.register_success_snippet()
            except:
                print(f'Failed to parse a method of snippet')
                log.register_fail_snippet()
                continue
        if i % 1000 == 0 and i > 0:
            print(f'Parsed {i} methods')

with open(OUTPUT, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')

log.save_log('log.json')
