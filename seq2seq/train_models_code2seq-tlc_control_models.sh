# Check if there are no arguments
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided. Please provide at least one argument."
    exit 1
fi

# Check if the first argument is empty
if [ -z "$1" ]; then
    echo "Error: The first argument is empty. Please provide a non-empty argument."
    exit 1
fi

path="$1"
echo "Path: $path"

seeds=(12345 789)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"
  #T5
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "t5-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text_tlc/seed_$seed/t5" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --prefix "Summarize Java: " \
    --seed $seed \
    --learning_rate 4e-5

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/t5/best_checkpoint" \
    --tokenizer_source "t5-base" \
    --tokenizer_target "t5-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --prefix "Summarize Java: "

  # bart
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "facebook/bart-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text_tlc/seed_$seed/bart" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/bart/best_checkpoint" \
    --tokenizer_source "facebook/bart-base" \
    --tokenizer_target "facebook/bart-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128

  #T5v1
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "google/t5-v1_1-small" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text_tlc/seed_$seed/t5v1" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --prefix "Summarize Java: " \
    --fp16 False \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/t5v1/best_checkpoint" \
    --tokenizer_source "google/t5-v1_1-small" \
    --tokenizer_target "google/t5-v1_1-small" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --prefix "Summarize Java: "

  # rand rand
  python train.py \
    --architecture "rand+rand" \
    --encoder "microsoft/codebert-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text_tlc/seed_$seed/rand66" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/rand66/best_checkpoint" \
    --tokenizer_source "microsoft/codebert-base" \
    --tokenizer_target "microsoft/codebert-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128

  python train.py \
    --architecture "rand+rand" \
    --encoder "microsoft/codebert-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text_tlc/seed_$seed/rand63" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --decoder_rand_layers 3 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/rand63/best_checkpoint" \
    --tokenizer_source "microsoft/codebert-base" \
    --tokenizer_target "microsoft/codebert-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128

  python train.py \
    --architecture "rand+rand" \
    --encoder "microsoft/codebert-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text_tlc/seed_$seed/rand33" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --decoder_rand_layers 3 \
    --encoder_rand_layers 3 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/rand33/best_checkpoint" \
    --tokenizer_source "microsoft/codebert-base" \
    --tokenizer_target "microsoft/codebert-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128
done