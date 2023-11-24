
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

seeds=(123 12345 789 72 93)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"

  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "t5-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text_tlc/seed_$seed/t5_fpfalse" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --prefix "Summarize Java: " \
    --seed $seed \
    --fp16 False \
    --learning_rate 4e-5

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/t5_fpfalse/best_checkpoint" \
    --tokenizer_source "t5-base" \
    --tokenizer_target "t5-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --prefix "Summarize Java: "
done