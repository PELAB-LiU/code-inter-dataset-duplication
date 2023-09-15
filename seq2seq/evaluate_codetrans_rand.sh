
python generate_predictions.py \
  --checkpoint "randrand_java2csharp/best_checkpoint" \
  --tokenizer_source "microsoft/codebert-base" \
  --tokenizer_target "microsoft/codebert-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512 \

