seeds=(123 72 93)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-large" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "code2text/seed_$seed/codet5large_prefix" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 32 \
    --patience 3 \
    --generation_max_length 128 \
    --prefix_tuning \
    --learning_rate 3e-4 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "code2text/seed_$seed/codet5large_prefix/best_checkpoint" \
    --base_model "Salesforce/codet5-large" \
    --tokenizer_source "Salesforce/codet5-large" \
    --tokenizer_target "Salesforce/codet5-large" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --prefix_tuning

  # codet5 small
  # codeT5 prefix
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-small" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "code2text/seed_$seed/codet5small_prefix" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 32 \
    --patience 3 \
    --generation_max_length 128 \
    --prefix_tuning \
    --learning_rate 3e-4 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "code2text/seed_$seed/codet5small_prefix/best_checkpoint" \
    --base_model "Salesforce/codet5-small" \
    --tokenizer_source "Salesforce/codet5-small" \
    --tokenizer_target "Salesforce/codet5-small" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --prefix_tuning
done
