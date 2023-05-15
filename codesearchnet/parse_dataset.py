import json
import pickle
import sys

import pandas as pd
from tqdm import tqdm

# setting path
sys.path.append('..')
from utils import get_tokens_from_snippet


def load_dataset(lang="java"):
    with open(f'{lang}_dedupe_definitions_v2.pkl', 'rb') as f:
        # Load the dictionary from the file
        my_dict = pickle.load(f)
    return pd.DataFrame(my_dict)


i = 0
with open('data.jsonl', 'w') as file:
    for language in ['java', 'python']:
        dataset = load_dataset(lang=language)
        for _, data in tqdm(dataset.iterrows()):
            try:
                json_string = json.dumps({"id_within_dataset": i,
                                          "snippet": data['function'],
                                          "tokens": get_tokens_from_snippet(data['function'],
                                                                            data['language']),
                                          "language": data['language'],
                                          "nl": data['docstring']})
            except:
                print(f'Failed to parse snippet {i}')
                continue
            file.write(json_string + '\n')
            i += 1

    print(f'Wrote {i} snippets to data.jsonl')
