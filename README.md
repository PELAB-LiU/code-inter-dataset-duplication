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

To upload the fine-tuning datasets to Hugging Face's hub, perform the following steps:

1. Compute inter-dataset duplication percentages and save representative data.
```shell
python dataset_overlapping.py --lang python --save_inter_representatives --compute_representatives
python dataset_overlapping.py --lang java --save_inter_representatives --compute_representatives
```

2. Upload the dataset to HF, considering inter-duplicate and representative information.
```shell
python upload_hf_dataset.py --data dataset_name/data.jsonl --inter dataset_name/interduplicates.json --rep dataset_name/representatives.json --hf_dataset hf_dir
```

To build and upload to HF our small pre-training datasets (leaky and non-leaky), perform the following steps:

1. Compute the samples that are included in CSN and are included in the fine-tuning datasets:
```shell
python get_pretraining.py --lang java
python get_pretraining.py --lang python
```

2. Filter those samples that contain encoding errors and that have no nl documentation associated.
```shell
python upload_csn_hf.py --first_stage
```

3. Perform the sampling strategy described in the paper and upload the dataset to HF.
```shell
python upload_csn_hf.py --hf_dataset PELAB-LiU/csn-small-interduplication --samples 100_000
```

The dataset used in our paper to pre-train the models can be found in https://huggingface.co/datasets/PELAB-LiU/csn-small-interduplication.
This dataset contains two splits: unbiased (non-leaky pre-training dataset) and biased (leaky pre-training dataset).


## Pre-training leaky and non-leaky LLMs

To pre-train the leaky and non-leaky LLMs, you can use the scripts of the folder `pretraining`.
```shell
python pretraining/train_mlm.py --dataset PELAB-LiU/csn-small-interduplication --split biased --checkpoint csn-small-biased-random-20 --base_model microsoft/unixcoder-base --initialize_random
python pretraining/train_mlm.py --dataset PELAB-LiU/csn-small-interduplication --split unbiased --checkpoint csn-small-unbiased-random-20 --base_model microsoft/unixcoder-base --initialize_random

python pretraining/train_bimodal_dual.py --dataset PELAB-LiU/csn-small-interduplication --split biased --checkpoint csn-small-biased-random-encoder-decoder-20 --base_model Salesforce/codet5-base
python pretraining/train_bimodal_dual.py --dataset PELAB-LiU/csn-small-interduplication --split unbiased --checkpoint csn-small-unbiased-random-encoder-decoder-20 --base_model Salesforce/codet5-base
```

The pre-trained models used in our paper are available at:
* (Leaky encoder) https://huggingface.co/PELAB-LiU/csn-small-biased-random-20
* (Non-leaky encoder) https://huggingface.co/PELAB-LiU/csn-small-unbiased-random-20
* (Leaky encoder-decoder) https://huggingface.co/PELAB-LiU/csn-small-biased-random-encoder-decoder-20
* (Non-leaky encoder-decoder) https://huggingface.co/PELAB-LiU/csn-small-unbiased-random-encoder-decoder-20

## Running fine-tuning procedures

These are the scripts used for fine-tuning our LLMs. They use the HF datasets: PELAB-LiU/tlc_interduplication,
PELAB-LiU/python-150_interduplication, and PELAB-LiU/codetrans-interduplication.

### Code summarization and code translation tasks

```shell
cd seq2seq
```

Code translation:
```shell
./script_b_ub/train_models_codetrans_biased.sh .
./script_b_ub/train_models_codetrans_unbiased.sh .

./script_b_ub/train_models_codetrans_biased_lora.sh .
./script_b_ub/train_models_codetrans_unbiased_lora.sh .

./script_b_ub/train_models_codetrans_biased_prefix.sh .
./script_b_ub/train_models_codetrans_unbiased_prefix.sh .
```

Code summarization using Python-150:
```shell
./script_b_ub/train_models_code2seq-python-150_biased_model.sh .
./script_b_ub/train_models_code2seq-python-150_unbiased_model.sh .

./script_b_ub/train_models_code2seq-python-150_biased_model_lora.sh .
./script_b_ub/train_models_code2seq-python-150_unbiased_model_lora.sh .

./script_b_ub/train_models_code2seq-python-150_biased_model_prefix.sh .
./script_b_ub/train_models_code2seq-python-150_unbiased_model_prefix.sh .
```

Code summarization using TLC:
```shell
./script_b_ub/train_models_code2seq-tlc_biased_model.sh .
./script_b_ub/train_models_code2seq-tlc_unbiased_model.sh .

./script_b_ub/train_models_code2seq-tlc_biased_model_lora.sh .
./script_b_ub/train_models_code2seq-tlc_unbiased_model_lora.sh .

./script_b_ub/train_models_code2seq-tlc_biased_model_prefix.sh .
./script_b_ub/train_models_code2seq-tlc_unbiased_model_prefix.sh .
```

To get the csvs (one per dataset) with the results, just run:
```shell
python extract_results_un_biased.py --folder ./code2text/ --task code2text --lang python --output code2text_un_biased.csv
python extract_results_un_biased.py --folder ./code2text_tlc/ --task code2text --lang java --output code2text_tlc_un_biased.csv
python extract_results_un_biased.py --folder ./codetrans/ --task codetrans --lang java --output code2trans_un_biased.csv
```

### Code search

First, go to the code search folder:
```shell
cd code_search
```

Then, run the following scripts:
```shell
./scripts/train_models_codesearch_biased.sh
./scripts/train_models_codesearch_unbiased.sh
./scripts/train_models_codesearch_layerwise_biased.sh
./scripts/train_models_codesearch_layerwise_unbiased.sh
```

Pre-trained models can also be trained for the layerwise setting
```shell
./scripts/train_models_codesearch_layerwise_pretrained.sh <model_name>
```
Replace `<model_name>` with the name of the pre-trained model in the paper, such as `microsoft/unixcoder-base`.

To get the csvs (one per dataset) with the results, just run:
```shell
python measure_performance.py --path ./results/ --output codesearch.csv
python measure_performance_layerwise.py --path ./results/ --output codesearch_layerwise.csv
```

## Case studies

More examples of samples that are fully remembered, partially remembered, and not remembered 
can be found in `seq2seq/case_studies.txt`.
