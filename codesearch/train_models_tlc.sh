
python train.py \
  --checkpoint "codebert_tlc.bin" \
  --data_path_hf "antolin/tlc_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train

python train.py \
  --checkpoint "codebert_tlc_peft.bin" \
  --data_path_hf "antolin/tlc_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --peft

python train.py \
  --checkpoint "roberta_tlc.bin" \
  --model_name_or_path "roberta-base" \
  --data_path_hf "antolin/tlc_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train

python train.py \
  --checkpoint "codebert_tlc_prefix.bin" \
  --data_path_hf "antolin/tlc_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --prefix_tuning

