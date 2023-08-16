# code-inter-dataset-duplication

Creation of empty database.
```shell
sqlite3 interduplication.db < schema.sql
```

## Download and index dataset

Download and file generation (`data.jsonl`) and other files.
```shell
cd dataset_name
./download.sh
python parse_dataset.py
```

Index in the database.
```shell
cd ..
python register_in_db.py --data dataset_name/data.jsonl --meta dataset_name/metadata.yaml
```

## Compute duplicates
```shell
python compute_duplicates.py --lang java|python
```

## Compute inter-duplication percentages

This computes the percentages of inter-duplication between codesearchnet and the rest of the datasets. It also saves a
json file for each dataset containing the snippets that are in codesearchnet and the given dataset.
```shell
python dataset_overlapping.py --lang java|python
```

## Running fine-tuning procedures

Code search (default over python-150):
```shell
python code_search.py 
```

## TODOs

- Python 150 take the original version and maybe preprocess comments
- Lists of datasets

