
python generate_predictions.py \
  --checkpoint "codet5_code2text_tlc/best_checkpoint" \
  --tokenizer_source "Salesforce/codet5-small" \
  --tokenizer_target "Salesforce/codet5-small" \
  --data_path_hf "antolin/tlc_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128 \
  --prefix "Summarize Java: " \
