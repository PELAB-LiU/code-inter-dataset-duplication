
python generate_predictions.py \
  --checkpoint "codebertrand_java2csharp/best_checkpoint" \
  --tokenizer_source "microsoft/codebert-base" \
  --tokenizer_target "microsoft/codebert-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512 \

echo "Full test set"
python evaluate_predictions.py --references codebertrand_java2csharp/best_checkpoint/references_full.txt \
  --predictions codebertrand_java2csharp/best_checkpoint/predictions_full.txt

echo "Without duplication"
python evaluate_predictions.py --references codebertrand_java2csharp/best_checkpoint/references_no_dup.txt \
  --predictions codebertrand_java2csharp/best_checkpoint/predictions_no_dup.txt