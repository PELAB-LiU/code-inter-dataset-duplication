import ast
import json
import os.path
import sys
from lib2to3 import refactor

from tqdm import tqdm

sys.path.append('..')
from utils import get_tokens_from_snippet, ParseLog


def convert_python2_to_python3(code):
    fixer_names = refactor.get_fixers_from_package('lib2to3.fixes')
    fixer = refactor.RefactoringTool(fixer_names)
    converted_code = fixer.refactor_string(code, '<string>')
    return str(converted_code)


def extract_method_snippets(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    code = convert_python2_to_python3(code)
    try:
        tree = ast.parse(code)
    except:
        tree = ast.parse(convert_python2_to_python3(code))

    method_snippets = []

    # Visit each node in the abstract syntax tree
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Extract the name and body of the function
            name = node.name
            body = ast.get_source_segment(code, node)

            # Append the method snippet to the list
            method_snippets.append((name, body))

    return method_snippets


PREFIX = 'py150_files'
OUTPUT = 'data.jsonl'

paths = {}
with open('py150_files/python50k_eval.txt', 'r') as file:
    paths["test"] = file.readlines()
    paths["test"] = [os.path.join(PREFIX, path).strip() for path in paths["test"]]
with open('py150_files/python100k_train.txt', 'r') as file:
    paths["train"] = file.readlines()
    paths["train"] = [os.path.join(PREFIX, path).strip() for path in paths["train"]]

all_data = []
i = 0
log = ParseLog()
for split in paths:
    for path in tqdm(paths[split], desc=f'Parsing {split} split'):
        try:
            methods = extract_method_snippets(path)
            log.register_success_file()
        except:
            log.register_fail_file()
            print(f'Failed to parse snippet {path}')
            continue
        for _, body in methods:
            try:
                all_data.append({"id_within_dataset": i,
                                 "snippet": body,
                                 "tokens": get_tokens_from_snippet(body, 'python'),
                                 "split_within_dataset": split})
                i += 1
                log.register_success_snippet()
            except:
                print(f'Failed to parse a method of snippet {path}')
                log.register_fail_snippet()
                continue

with open(OUTPUT, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')

log.save_log('log.json')

