
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-small" \
  --data_path_hf "antolin/tlc_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "codet5_code2text_tlc" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "bleu-code2text-cxg" \
  --prefix "Summarize Java: "
