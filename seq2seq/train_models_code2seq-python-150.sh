# codeT5 lora
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "codet5_code2text_python-150_peft" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "bleu-code2text-cxg" \
  --peft

python generate_predictions.py \
  --checkpoint "codet5_code2text_python-150_peft/best_checkpoint" \
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



# codeT5
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "codet5_code2text_python-150" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "bleu-code2text-cxg"

python generate_predictions.py \
  --checkpoint "codet5_code2text_python-150/best_checkpoint" \
  --tokenizer_source "Salesforce/codet5-base" \
  --tokenizer_target "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128


#T5
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "t5-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "t5_code2text_python-150" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "bleu-code2text-cxg" \
  --prefix "Summarize Python: "

python generate_predictions.py \
  --checkpoint "t5_code2text_python-150/best_checkpoint" \
  --tokenizer_source "t5-base" \
  --tokenizer_target "t5-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128 \
  --prefix "Summarize Python: "
