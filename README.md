# code-inter-dataset-duplication

Creation of empty database.
```shell
sqlite3 interduplication.db < schema.sql
sqlite3 bigclonebench.db < schema.sql
```

## BigCloneBench

Download and file generation (`data.jsonl`) and other files (note that this version comes from CodeXGlue).
```shell
cd bigclonebench
./download.sh
python parse_dataset.py
```

Indexing in the databases.
```shell
cd ..
python register_in_db.py --data bigclonebench/data.jsonl --meta bigclonebench/metadata.yaml
python register_in_db.py --data bigclonebench/data.jsonl --meta bigclonebench/metadata.yaml --db bigclonebench.db
```

Test method.
```shell
python compute_duplicates.py --lang java --db bigclonebench.db
python bigclonebench/test_duplication_method.py 
```


## Java-(small|medium|large)

```shell
cd java-small
./download.sh
python parse_dataset.py
```

Indexing in the databases.
```shell
cd ..
python register_in_db.py --data java-small/data.jsonl --meta java-small/metadata.yaml
```


## CodeSearchNet
    
```shell
cd codesearchnet
./download.sh
python parse_dataset.py
```

Indexing in the databases.
```shell
cd ..
python register_in_db.py --data codesearchnet/data.jsonl --meta codesearchnet/metadata.yaml 
```


## Compute duplicates
```shell
python compute_duplicates.py --lang java
```

