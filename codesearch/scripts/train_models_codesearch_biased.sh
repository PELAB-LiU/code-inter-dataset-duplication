
seeds=(1 2 3 4 5 6 7 8 9 10)

model_name="csn-small-biased-random-20"
hf_model_name="antolin/$model_name"

for seed in "${seeds[@]}";
do
echo "Running seed $seed"

# biased_roberta_small LORA
python train.py \
  --checkpoint "results/encoder/$seed/lora/$model_name-lora.bin" \
  --model_name_or_path "$hf_model_name" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 10 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --lora \
  --seed $seed \
  --learning_rate 3e-4

# biased_roberta_small FF
python train.py \
  --checkpoint "results/encoder/$seed/ff/$model_name-ff.bin" \
  --model_name_or_path "$hf_model_name" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 10 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --seed $seed \
  --do_train

# biased_roberta_small PT
python train.py \
  --checkpoint "results/encoder/$seed/prefix/$model_name-prefix.bin" \
  --model_name_or_path "$hf_model_name" \
  --data_path_hf "antolin/python-150_interduplication" \
  --tokens_column "tokens" \
  --nl_column "nl" \
  --num_train_epochs 10 \
  --max_code_len 256 \
  --max_nl_len 128 \
  --do_train \
  --prefix_tuning \
  --seed $seed \
  --learning_rate 1e-5
  
  done