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

## Upload to HF

This works for all dataset less than BigCloneBench. For BigCloneBench, see the next section.
```shell
python upload_hf_dataset.py --data dataset_name/data.jsonl --inter dataset_name/interduplicates.json --rep dataset_name/representatives.json --hf_dataset hf_dir
```

### BigCloneBench case

```shell
cd bigcodebench
python compute_problematic_pairs_upload_hf.py
```

## Running fine-tuning procedures

```shell
cd seq2seq
```

Code translation:
```shell
./train_codetrans.sh
./evaluate_codetrans.sh
```

Code documentation:


## TODOs

- Scripts training
- Scripts upload to hf

