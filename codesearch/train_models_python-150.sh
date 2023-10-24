# # CODEBERT LORA
# python train.py \
#   --checkpoint "codebert_python-150_lora.bin" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train \
#   --lora \
#   --learning_rate 3e-4


# # UNIXCODER LORA
# python train.py \
#   --checkpoint "unixcoder_python-150_lora.bin" \
#   --model_name_or_path "microsoft/unixcoder-base" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train \
#   --lora \
#   --learning_rate 3e-4
  
# # GRAPHCODEBERT LORA
# python train.py \
#   --checkpoint "graphcodebert_python-150_lora.bin" \
#   --model_name_or_path "microsoft/graphcodebert-base" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train \
#   --lora \
#   --learning_rate 3e-4

# # CODEBERT FF
# python train.py \
#   --checkpoint "codebert_python-150.bin" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train


# # UNIXCODER FF
# python train.py \
#   --checkpoint "unixcoder_python-150.bin" \
#   --model_name_or_path "microsoft/unixcoder-base" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train
  
# # GRAPHCODEBERT FF
# python train.py \
#   --checkpoint "graphcodebert_python-150.bin" \
#   --model_name_or_path "microsoft/graphcodebert-base" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train \

  
  
# # CODEBERT PT
# python train.py \
#   --checkpoint "codebert_python-150_prefix.bin" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train \
#   --prefix_tuning \
#   --learning_rate 1e-5


  
# # UNIXCODER PT
# python train.py \
#   --checkpoint "unixcoder_python-150_prefix.bin" \
#   --model_name_or_path "microsoft/unixcoder-base" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train \
#   --prefix_tuning \
#   --learning_rate 1e-5
  
# # GRAPHCODEBERT PT
# python train.py \
#   --checkpoint "graphcodebert_python-150_prefix.bin" \
#   --model_name_or_path "microsoft/graphcodebert-base" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 5 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train \
#   --prefix_tuning \
#   --learning_rate 1e-5


# RANDOM
# python train.py \
#   --checkpoint "random_python-150.bin" \
#   --data_path_hf "antolin/python-150_interduplication" \
#   --tokens_column "tokens" \
#   --nl_column "nl" \
#   --num_train_epochs 10 \
#   --max_code_len 256 \
#   --max_nl_len 128 \
#   --do_train \
#   --is_baseline

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
  --do_train