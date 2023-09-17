
python train.py \
  --checkpoint "rand_tlc.bin" \
  --data_path_hf "antolin/tlc_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --is_baseline \

python train.py \
  --checkpoint "rand_tlc.bin" \
  --data_path_hf "antolin/tlc_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 5 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --is_baseline \


