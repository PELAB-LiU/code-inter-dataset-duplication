import json
import tokenize
from io import BytesIO

DESCRIPTIONS_FILE = 'descriptions.txt'
SNIPPETS_FILE = 'snippets.txt'
OUTPUT = 'data.jsonl'

FILTER_TOKENS = [tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING]


def get_tokens(code):
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    result = []
    for token in tokens:
        if token.type not in FILTER_TOKENS:
            result.append(token.string)
    return result


all_data = []
i = 0
with open(DESCRIPTIONS_FILE, 'r') as file1, open(SNIPPETS_FILE, 'r') as file2:
    for description, code in zip(file1, file2):
        all_data.append({"id_within_dataset": i,
                         "snippet": code,
                         "tokens": get_tokens(code),
                         "nl": description})
        i += 1

with open(OUTPUT, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')
