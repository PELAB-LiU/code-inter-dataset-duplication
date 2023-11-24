seeds=(123)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"
  # codeT5 lora
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-base" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "code2text/seed_$seed/codet5_lora" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --lora \
    --learning_rate 3e-4 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "code2text/seed_$seed/codet5_lora/best_checkpoint" \
    --base_model "Salesforce/codet5-base" \
    --tokenizer_source "Salesforce/codet5-base" \
    --tokenizer_target "Salesforce/codet5-base" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --lora

  # codeT5 prefix tuning
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-base" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "code2text/seed_$seed/codet5_prefix" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 32 \
    --patience 3 \
    --generation_max_length 128 \
    --prefix_tuning \
    --learning_rate 3e-4 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "code2text/seed_$seed/codet5_prefix/best_checkpoint" \
    --base_model "Salesforce/codet5-base" \
    --tokenizer_source "Salesforce/codet5-base" \
    --tokenizer_target "Salesforce/codet5-base" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --prefix_tuning
done