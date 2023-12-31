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
  # roberta
  python train.py \
    --output_dir "$path/code_clone/seed_$seed/roberta" \
    --encoder "roberta-base" \
    --do_train \
    --seed $seed

  # bert
  python train.py \
    --output_dir "$path/code_clone/seed_$seed/bert" \
    --encoder "bert-base-uncased" \
    --do_train \
    --seed $seed

  # mbert
  python train.py \
    --output_dir "$path/code_clone/seed_$seed/mbert" \
    --encoder "bert-base-multilingual-uncased" \
    --do_train \
    --seed $seed

  # rand
  python train.py \
    --output_dir "$path/code_clone/seed_$seed/rand6" \
    --is_baseline \
    --layers 6 \
    --do_train \
    --seed $seed

  python train.py \
    --output_dir "$path/code_clone/seed_$seed/rand3" \
    --is_baseline \
    --layers 3 \
    --do_train \
    --seed $seed

  python train.py \
    --output_dir "$path/code_clone/seed_$seed/rand1" \
    --is_baseline \
    --layers 1 \
    --do_train \
    --seed $seed
done