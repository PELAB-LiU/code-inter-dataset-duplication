seeds=(123 72 93)

for seed in "${seeds[@]}";
do
  echo "Seed: $seed"
  # codeT5 lora
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "codetrans/seed_$seed/codet5_lora" \
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
    --lora \
    --learning_rate 3e-4 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "codetrans/seed_$seed/codet5_lora/best_checkpoint" \
    --base_model "Salesforce/codet5-base" \
    --tokenizer_source "Salesforce/codet5-base" \
    --tokenizer_target "Salesforce/codet5-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512 \
    --lora

  # codeT5
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "codetrans/seed_$seed/codet5_ff" \
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
    --checkpoint "codetrans/seed_$seed/codet5_ff/best_checkpoint" \
    --tokenizer_source "Salesforce/codet5-base" \
    --tokenizer_target "Salesforce/codet5-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512

  # codeT5 prefix
  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --output_dir "codetrans/seed_$seed/codet5_prefix" \
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
    --prefix_tuning \
    --learning_rate 3e-4 \
    --seed $seed

  python generate_predictions.py \
    --checkpoint "codetrans/seed_$seed/codet5_prefix/best_checkpoint" \
    --base_model "Salesforce/codet5-base" \
    --tokenizer_source "Salesforce/codet5-base" \
    --tokenizer_target "Salesforce/codet5-base" \
    --data_path_hf "antolin/codetrans_interduplication" \
    --source_column "snippet" \
    --target_column "cs" \
    --max_length_source 512 \
    --max_length_target 512 \
    --prefix_tuning
  done
