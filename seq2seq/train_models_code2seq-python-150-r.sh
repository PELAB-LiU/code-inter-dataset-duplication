
current_value=64

# Set the maximum value you want to reach
max_value=256  # You can change this to your desired maximum value

# Start the for loop
while [ $current_value -le $max_value ]
do

  echo "r=$current_value"

  python train.py \
    --architecture "encoder-decoder" \
    --encoder_decoder "Salesforce/codet5-base" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --output_dir "codet5_code2text_python-150_peft_r$current_value" \
    --num_train_epochs 10 \
    --max_length_source 256 \
    --max_length_target 128 \
    --patience 3 \
    --generation_max_length 128 \
    --metric_for_best_model "bleu-code2text-cxg" \
    --peft \
    --r $current_value

  python generate_predictions.py \
    --checkpoint "codet5_code2text_python-150_peft_r$current_value/best_checkpoint" \
    --base_model "Salesforce/codet5-base" \
    --tokenizer_source "Salesforce/codet5-base" \
    --tokenizer_target "Salesforce/codet5-base" \
    --data_path_hf "antolin/python-150_interduplication" \
    --source_column "tokens" \
    --is_split_source \
    --target_column "nl" \
    --max_length_source 256 \
    --max_length_target 128 \
    --peft


  # Double the current value
  current_value=$((current_value * 2))
done


