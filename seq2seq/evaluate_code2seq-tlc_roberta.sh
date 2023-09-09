
python generate_predictions.py \
  --checkpoint "robertarand_code2text_tlc/best_checkpoint" \
  --tokenizer_source "roberta-base" \
  --tokenizer_target "roberta-base" \
  --data_path_hf "antolin/tlc_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128 \
