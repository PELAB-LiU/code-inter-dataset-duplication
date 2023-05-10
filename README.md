# code-inter-dataset-duplication

Creation of empty database.
```shell
sqlite3 interduplication.db < schema.sql
```

## BigCloneBench

Download and file generation (`data.jsonl`) and other files.
```shell
cd bigclonebench
./download.sh
python parse_dataset.py
```

Indexing in the database.
```shell
cd ..
python register_in_db.py --data bigclonebench/data.jsonl --meta bigclonebench/metadata.yaml
```

## CodeSearchNet
    
```shell
cd codesearchnet
./download.sh
```


