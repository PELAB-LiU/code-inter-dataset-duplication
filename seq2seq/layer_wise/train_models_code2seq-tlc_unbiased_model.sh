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
seed=11

tellies=(1 2 3 4 5 6 7 8 9 10 11)

for t in "${tellies[@]}";
do
  echo "Seed: $seed"
  echo "tlc unbiased model FF telly $t decoder"
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "antolin/csn-small-unbiased-random-encoder-decoder-20" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "$path/code2text_tlc/seed_$seed/random_unbiased_decoder_telly_$t" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --seed $seed \
    --telly $t

  python generate_predictions.py \
    --checkpoint "$path/code2text_tlc/seed_$seed/random_unbiased_decoder_telly_$t/best_checkpoint" \
    --tokenizer_source "antolin/csn-small-unbiased-random-encoder-decoder-20" \
    --tokenizer_target "antolin/csn-small-unbiased-random-encoder-decoder-20" \
    --data_path_hf "antolin/tlc_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128

done