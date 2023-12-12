# Inter-dataset code duplication

This repository contains the code for the paper "On Inter-dataset Code Duplication 
and Data Leakage in Large Language Models".

## Creation of the inter-duplication database

Here we describe the steps to create the inter-dataset code duplication database.

1. Create an empty database using the provided SQL schema.
```shell
sqlite3 interduplication.db < schema.sql
```

The schema of the target database is:
```sqlite
CREATE TABLE datasets (
	id TEXT PRIMARY KEY,
	url TEXT NOT NULL
);

CREATE TABLE snippets (
	id integer primary key autoincrement,
	snippet TEXT NOT NULL,
	dataset TEXT NOT NULL,
	language TEXT NOT NULL,
	tokens TEXT NOT NULL,
	id_within_dataset integer NOT NULL,
	split_within_dataset TEXT,
	CONSTRAINT fk_dataset
        FOREIGN KEY (dataset)
        REFERENCES datasets (id)
        ON DELETE CASCADE
);

CREATE TABLE duplicates (
    id integer primary key autoincrement,
    snippet1 INT NOT NULL,
    snippet2 INT NOT NULL,
    CONSTRAINT fk_snippet1
        FOREIGN KEY (snippet1)
        REFERENCES snippets (id)
        ON DELETE CASCADE,
    CONSTRAINT fk_snippet2
        FOREIGN KEY (snippet2)
        REFERENCES snippets (id)
        ON DELETE CASCADE
);
```

2. Download and parse datasets for code duplication analysis.
```shell
cd dataset_name
./download.sh
python parse_dataset.py
```

3. Index the dataset in the database.
```shell
cd ..
python register_in_db.py --data dataset_name/data.jsonl --meta dataset_name/metadata.yaml
```

4. Compute code duplicates for the specified programming languages.
```shell
python compute_duplicates.py --lang java
python compute_duplicates.py --lang python
```

5. Compute the percentages of inter-duplication between CodeSearchNet and other datasets.
```shell
python dataset_overlapping.py --lang java|python
```

## Upload datasets to HF

To upload datasets to Hugging Face's hub, perform the following steps:

1. Compute inter-dataset duplication percentages and save representative data.
```shell
python dataset_overlapping.py --lang python --save_inter_representatives --compute_representatives
python dataset_overlapping.py --lang java --save_inter_representatives --compute_representatives
```

2. Upload the dataset to HF, considering inter-duplicate and representative information.
```shell
python upload_hf_dataset.py --data dataset_name/data.jsonl --inter dataset_name/interduplicates.json --rep dataset_name/representatives.json --hf_dataset hf_dir
```

## Running fine-tuning procedures

These are the scripts used for fine-tuning LLMs. They use the HF datasets: antolin/tlc_interduplication,
antolin/python-150_interduplication, and antolin/codetrans-interduplication.

### Code summarization and code translation tasks

```shell
cd seq2seq
```

Code translation (CodeT5-base and control models):
```shell
./train_models_codetrans.sh .
./train_models_codetrans_control_models.sh .
```

Code summarization using Python-150 and TLC (CodeT5-small, base, large, control models, lora, and prefix tuning):

```shell
./train_models_code2seq-python-150.sh .
./train_models_code2seq-python-150_control_models.sh .
./train_models_code2seq-python-150_peft.sh .

./train_models_code2seq-tlc.sh .
./train_models_code2seq-tlc_control_models.sh .
./train_models_code2seq-tlc_peft.sh .
```

### Code search

Code search using Python-150 (CodeBERT, GraphCodeBERT, UnixCoder, control models, lora, and prefix tuning):
```shell
cd codesearch
./train_models_python-150.sh 
```

Layer-wise analysis:
```shell
./train_models_python-150_telly.sh 
```
