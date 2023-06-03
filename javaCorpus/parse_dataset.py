import ast
import json
import os.path
import sys
from lib2to3 import refactor

from tqdm import tqdm

sys.path.append('..')
from utils import get_tokens_from_snippet, get_methods_java


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

file_contents = []
with open(f'{PREFIX}/test.txt', 'r') as file:
    file_contents += file.readlines()
with open(f'{PREFIX}/dev.txt', 'r') as file:
    file_contents += file.readlines()
with open(f'{PREFIX}/train.txt', 'r') as file:
    file_contents += file.readlines()

all_data = []
i = 0
for content in tqdm(file_contents):
    content = ' '.join(content.split(' ')[1:-1])
    content = add_new_lines(content)
    try:
        methods = get_methods_java(content)
    except:
        print(f'Failed to parse snippet')
        continue
    for body in methods:
        try:
            all_data.append({"id_within_dataset": i,
                             "snippet": body,
                             "tokens": get_tokens_from_snippet(body, 'java')})
            i += 1
        except:
            print(f'Failed to parse a method of snippet')
            continue
    if i % 1000 == 0 and i > 0:
        print(f'Parsed {i} methods')

with open(OUTPUT, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')
