
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

seeds=(123 72 93 12345 789)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"
  #T5
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "t5-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "$path/codetrans/seed_$seed/t5_fpfalse" \
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
    --seed $seed \
    --learning_rate 4e-5 \
    --fp16 False


  python generate_predictions.py \
    --checkpoint "$path/codetrans/seed_$seed/t5_fpfalse/best_checkpoint" \
    --tokenizer_source "t5-base" \
    --tokenizer_target "t5-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512

  #T5 v1
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "google/t5-v1_1-small" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "$path/codetrans/seed_$seed/t5v1" \
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
    --fp16 False \
    --seed $seed


  python generate_predictions.py \
    --checkpoint "$path/codetrans/seed_$seed/t5v1/best_checkpoint" \
    --tokenizer_source "google/t5-v1_1-small" \
    --tokenizer_target "google/t5-v1_1-small" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512


  # bart
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "facebook/bart-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "$path/codetrans/seed_$seed/bart" \
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
    --checkpoint "$path/codetrans/seed_$seed/bart/best_checkpoint" \
    --tokenizer_source "facebook/bart-base" \
    --tokenizer_target "facebook/bart-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512

  python train.py \
    --architecture "rand+rand" \
    --encoder "microsoft/codebert-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "$path/codetrans/seed_$seed/rand66" \
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
    --checkpoint "$path/codetrans/seed_$seed/rand66/best_checkpoint" \
    --tokenizer_source "microsoft/codebert-base" \
    --tokenizer_target "microsoft/codebert-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512

  python train.py \
    --architecture "rand+rand" \
    --encoder "microsoft/codebert-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "$path/codetrans/seed_$seed/rand63" \
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
    --decoder_rand_layers 3 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/codetrans/seed_$seed/rand63/best_checkpoint" \
    --tokenizer_source "microsoft/codebert-base" \
    --tokenizer_target "microsoft/codebert-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512


  python train.py \
    --architecture "rand+rand" \
    --encoder "microsoft/codebert-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "$path/codetrans/seed_$seed/rand33" \
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
    --decoder_rand_layers 3 \
    --encoder_rand_layers 3 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "$path/codetrans/seed_$seed/rand33/best_checkpoint" \
    --tokenizer_source "microsoft/codebert-base" \
    --tokenizer_target "microsoft/codebert-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512
  done