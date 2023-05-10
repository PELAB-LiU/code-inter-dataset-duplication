import glob
import pickle
from pathlib import Path

import pandas as pd
from datasets import load_dataset, concatenate_datasets


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
