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

seeds=(123 12345 72 789 93)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"

  # lora
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-large" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text/seed_$seed/codet5large_lora" \
    --num_train_epochs 5 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --seed $seed \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --lora \
    --learning_rate 3e-4

  python generate_predictions.py \
    --checkpoint "$path/code2text/seed_$seed/codet5large_lora/best_checkpoint" \
    --tokenizer_source "Salesforce/codet5-large" \
    --tokenizer_target "Salesforce/codet5-large" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --lora \
    --base_model "Salesforce/codet5-large"

  # prefix
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-large" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text/seed_$seed/codet5large_prefix" \
    --num_train_epochs 5 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --seed $seed \
    --prefix_tuning \
    --learning_rate 3e-4

  python generate_predictions.py \
    --checkpoint "$path/code2text/seed_$seed/codet5large_prefix/best_checkpoint" \
    --tokenizer_source "Salesforce/codet5-large" \
    --tokenizer_target "Salesforce/codet5-large" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --prefix_tuning \
    --base_model "Salesforce/codet5-large"
done