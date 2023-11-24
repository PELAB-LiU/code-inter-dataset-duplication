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

seeds=(123 72 93)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/codet5small_ff/best_checkpoint" \
    --tokenizer_source "Salesforce/codet5-small" \
    --tokenizer_target "Salesforce/codet5-small" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/codet5_ff/best_checkpoint" \
    --tokenizer_source "Salesforce/codet5-base" \
    --tokenizer_target "Salesforce/codet5-base" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/codet5large_ff/best_checkpoint" \
    --tokenizer_source "Salesforce/codet5-large" \
    --tokenizer_target "Salesforce/codet5-large" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128

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