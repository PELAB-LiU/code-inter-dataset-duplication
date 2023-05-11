import glob
import json
import pickle
from pathlib import Path

import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def load_local_dataset(lang="all", path="data"):
    """
    Load a local dataset from the downloaded Kaggle dataset.

    Args:
        lang (str): The language to be used for the dataset.
        path (str, optional): Path to the downloaded dataset. Defaults to "data".

    Returns:
        Dataset: dataset loaded from local files
    """
    path = Path(path)

    if lang != "all":
        # Read the downloaded dataset
        path = path / lang / lang / "final/jsonl"
        dataset = load_dataset(
            "json",
            data_files={
                "train": glob.glob(path.as_posix() + "/train/*.jsonl"),
                "validation": glob.glob(path.as_posix() + "/valid/*.jsonl"),
                "test": glob.glob(path.as_posix() + "/test/*.jsonl"),
            },
        )
    else:
        train_files = glob.glob(path.as_posix() + "/**/train/*.jsonl", recursive=True)
        valid_files = glob.glob(path.as_posix() + "/**/valid/*.jsonl", recursive=True)
        test_files = glob.glob(path.as_posix() + "/**/test/*.jsonl", recursive=True)
        dataset = load_dataset(
            "json",
            data_files={
                "train": train_files,
                "validation": valid_files,
                "test": test_files,
            },
        )

    return dataset


def load_unimodal(lang="java"):
    with open(f'data/java/{lang}_dedupe_definitions_v2.pkl', 'rb') as f:
        # Load the dictionary from the file
        my_dict = pickle.load(f)
    return pd.DataFrame(my_dict)


def load_datasets(lang="java"):
    multimodal = load_local_dataset(lang=lang)
    multimodal = concatenate_datasets([multimodal['train'],
                                       multimodal['validation'],
                                       multimodal['test']]).to_pandas()
    unimodal = load_unimodal(lang=lang)
    return multimodal, unimodal


multimodal, unimodal = load_datasets(lang="java")

with open('data.jsonl', 'w') as file:
    i = 0
    for _, data in tqdm(multimodal.iterrows()):
        json_string = json.dumps({"id_within_dataset": i,
                                  "snippet": data['code'],
                                  "tokens": data['code_tokens'].tolist(),
                                  "nl": data['docstring'],
                                  "nl_tokens": data['docstring_tokens'].tolist(),
                                  "language": data['language']})
        i += 1
        file.write(json_string + '\n')
    for _, data in tqdm(unimodal.iterrows()):
        json_string = json.dumps({"id_within_dataset": i,
                                  "snippet": data['function'],
                                  "tokens": data['function_tokens'],
                                  "language": data['language']})
        file.write(json_string + '\n')
        i += 1

print(f'Wrote {i} snippets to data.jsonl')
