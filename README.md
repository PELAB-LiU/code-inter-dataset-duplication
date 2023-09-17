



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

### Seq2seq tasks

```shell
cd seq2seq
```

Code translation:
```shell
./train_codetrans.sh
./evaluate_codetrans.sh
./train_codetrans_rand.sh
./evaluate_codetrans_rand.sh
python sentence_bleu_dc.py --data_folder codebertrand_java2csharp/best_checkpoint/ --task codetrans
python sentence_bleu_dc.py --data_folder randrand_java2csharp/best_checkpoint/ --task codetrans
```

Code documentation (tlc):
```shell
./train_code2seq-tlc.sh
./evaluate_code2seq-tlc.sh
./train_code2seq-tlc_rand.sh
./evaluate_code2seq-tlc_rand.sh
python sentence_bleu_dc.py --data_folder codebertrand_code2text_tlc/best_checkpoint/ --task code2text
python sentence_bleu_dc.py --data_folder randrand_code2text_tlc/best_checkpoint/ --task code2text
```

Code documentation (python-150):

```shell
./train_code2seq-python-150.sh
./evaluate_code2seq-python-150.sh
./train_code2seq-python-150_rand.sh
./evaluate_code2seq-python-150_rand.sh
python sentence_bleu_dc.py --data_folder codebertrand_code2text_python-150/best_checkpoint/ --task code2text
python sentence_bleu_dc.py --data_folder randrand_code2text_python-150/best_checkpoint/ --task code2text
```

### Code search

```shell
cd codesearch
```

TLC dataset

```shell
./train_tlc.sh 
```

Python-150 dataset

```shell
./train_python-150.sh 
```
