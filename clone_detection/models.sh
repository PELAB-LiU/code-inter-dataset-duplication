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

seeds=(12345 789 123 72 93)

for seed in "${seeds[@]}";
do
  echo "Loop: $seed"
  python train.py \
    --output_dir "$path/code_clone/seed_$seed/graphcodebert" \
    --encoder "microsoft/graphcodebert-base" \
    --do_train \
    --seed $seed

  python train.py \
    --output_dir "$path/code_clone/seed_$seed/codebert" \
    --encoder "microsoft/codebert-base" \
    --do_train \
    --seed $seed

  python train.py \
    --output_dir "$path/code_clone/seed_$seed/unixcoder" \
    --encoder "microsoft/unixcoder-base" \
    --do_train \
    --seed $seed
done