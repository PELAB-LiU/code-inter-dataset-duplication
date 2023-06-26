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

