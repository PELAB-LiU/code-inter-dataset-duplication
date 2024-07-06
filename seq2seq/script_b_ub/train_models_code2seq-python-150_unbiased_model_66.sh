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

seeds=(11 12 13 14 15 16 17 18 19 20)

for seed in "${seeds[@]}";
do
  echo "l66 unbiased python-150 Seed: $seed"
  echo "antolin/csn-small-unbiased-random-encoder-decoder-20-l66"
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "antolin/csn-small-unbiased-random-encoder-decoder-20-l66" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text/seed_$seed/random_unbiased_l66" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/code2text/seed_$seed/random_unbiased_l66/best_checkpoint" \
    --tokenizer_source "antolin/csn-small-unbiased-random-encoder-decoder-20-l66" \
    --tokenizer_target "antolin/csn-small-unbiased-random-encoder-decoder-20-l66" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128

done