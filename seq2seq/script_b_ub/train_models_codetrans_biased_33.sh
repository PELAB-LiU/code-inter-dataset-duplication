
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

seeds=(1 2 3 4 5 6 7 8 9 10)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"
  echo "biased codetrans l33"
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "antolin/csn-small-biased-random-encoder-decoder-20-l33" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "$path/codetrans/seed_$seed/random_biased_l33" \
    --max_length_source 512 \
    --max_length_target 512 \
    --num_train_epochs 150 \
    --patience 3 \
    --generation_max_length 512 \
    --save_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --max_steps 15000 \
    --per_device_train_batch_size 16 \
    --save_steps 5000 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/codetrans/seed_$seed/random_biased_l33/best_checkpoint" \
    --tokenizer_source "antolin/csn-small-biased-random-encoder-decoder-20-l33" \
    --tokenizer_target "antolin/csn-small-biased-random-encoder-decoder-20-l33" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512

  done
