# Inter-dataset code duplication

This repository contains the code for the paper "On Inter-dataset Code Duplication 
and Data Leakage in Large Language Models".

## Creation of the inter-duplication database

Here we describe the steps to create the inter-dataset code duplication database.

1. Create an empty database using the provided SQL schema.
```shell
sqlite3 interduplication.db < schema.sql
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

### Seq2seq tasks

```shell
cd seq2seq
```

Code translation:
```shell
./train_models_codetrans.sh
python sentence_bleu_dc.py --data_folder t5_java2csharp/best_checkpoint/ --task codetrans
python sentence_bleu_dc.py --data_folder codet5_java2csharp/best_checkpoint/ --task codetrans
python sentence_bleu_dc.py --data_folder codet5_java2csharp_peft/best_checkpoint/ --task codetrans
```

Code summarization (python-150):

```shell
./train_models_code2seq-python-150.sh
python sentence_bleu_dc.py --data_folder t5_code2text_python-150/best_checkpoint/ --task code2text
python sentence_bleu_dc.py --data_folder codet5_code2text_python-150/best_checkpoint/ --task code2text
python sentence_bleu_dc.py --data_folder t5_code2text_python-150_peft/best_checkpoint/ --task code2text
```

### Code search

```shell
cd codesearch
./train_models_python-150.sh 
```
