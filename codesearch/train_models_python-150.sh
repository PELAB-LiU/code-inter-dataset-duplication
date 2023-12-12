
seed=123

# UNIXCODER LORA
python train.py \
  --checkpoint "unixcoder_python-150_lora.bin" \
  --model_name_or_path "microsoft/unixcoder-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --lora \
  --seed $seed \
  --learning_rate 3e-4
  
# GRAPHCODEBERT LORA
python train.py \
  --checkpoint "graphcodebert_python-150_lora.bin" \
  --model_name_or_path "microsoft/graphcodebert-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --lora \
  --seed $seed \
  --learning_rate 3e-4

# CODEBERT FF
python train.py \
  --checkpoint "codebert_python-150.bin" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --do_train


# UNIXCODER FF
python train.py \
  --checkpoint "unixcoder_python-150.bin" \
  --model_name_or_path "microsoft/unixcoder-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --do_train
  
# GRAPHCODEBERT FF
python train.py \
  --checkpoint "graphcodebert_python-150.bin" \
  --model_name_or_path "microsoft/graphcodebert-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --do_train

  
# UNIXCODER PT
python train.py \
  --checkpoint "unixcoder_python-150_prefix.bin" \
  --model_name_or_path "microsoft/unixcoder-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --prefix_tuning \
  --seed $seed \
  --learning_rate 1e-5
  
# GRAPHCODEBERT PT
python train.py \
  --checkpoint "graphcodebert_python-150_prefix.bin" \
  --model_name_or_path "microsoft/graphcodebert-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --prefix_tuning \
  --seed $seed \
  --learning_rate 1e-5


# Control models
# ROBERTA
python train.py \
  --checkpoint "roberta_python-150.bin" \
  --model_name_or_path "roberta-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --do_train
  
# Bert
python train.py \
  --checkpoint "bert_python-150.bin" \
  --model_name_or_path "bert-base-uncased" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --do_train

# Multilingual Bert
python train.py \
  --checkpoint "bert_multilingual_python-150.bin" \
  --model_name_or_path "bert-base-multilingual-uncased" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --do_train

# Randomly initialized
python train.py \
  --checkpoint "bert_1_layer_python-150.bin" \
  --model_name_or_path "bert-base-uncased" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --num_layers 1\
  --is_baseline \
  --num_train_epochs 20 \
  --do_train 
  
python train.py \
  --checkpoint "bert_3_layer_python-150.bin" \
  --model_name_or_path "bert-base-uncased" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --num_layers 3\
  --is_baseline \
  --num_train_epochs 20 \
  --do_train \
  
  
  
python train.py \
  --checkpoint "bert_6_layer_python-150.bin" \
  --model_name_or_path "bert-base-uncased" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --num_layers 6\
  --is_baseline \
  --num_train_epochs 20 \
  --do_train 