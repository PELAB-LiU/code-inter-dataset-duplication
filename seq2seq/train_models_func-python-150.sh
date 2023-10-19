# codeT5 lora
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_func_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "func_name" \
  --output_dir "codet5_func_python-150_peft" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 32 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "f1_subtoken" \
  --peft

python generate_predictions.py \
  --checkpoint "codet5_func_python-150_peft/best_checkpoint" \
  --base_model "Salesforce/codet5-base" \
  --tokenizer_source "Salesforce/codet5-base" \
  --tokenizer_target "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_func_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "func_name" \
  --max_length_source 256 \
  --max_length_target 32 \
  --peft

# codet5

python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_func_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "func_name" \
  --output_dir "codet5_func_python-150" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 32 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "f1_subtoken"

python generate_predictions.py \
  --checkpoint "codet5_func_python-150/best_checkpoint" \
  --base_model "Salesforce/codet5-base" \
  --tokenizer_source "Salesforce/codet5-base" \
  --tokenizer_target "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_func_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "func_name" \
  --max_length_source 256 \
  --max_length_target 32


# T5
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "t5-base" \
  --data_path_hf "antolin/python-150_func_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "func_name" \
  --output_dir "t5_func_python-150" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 32 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "f1_subtoken"

python generate_predictions.py \
  --checkpoint "t5_func_python-150/best_checkpoint" \
  --base_model "t5-base" \
  --tokenizer_source "t5-base" \
  --tokenizer_target "t5-base" \
  --data_path_hf "antolin/python-150_func_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "func_name" \
  --max_length_source 256 \
  --max_length_target 32

# codeT5 prefix tuning
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_func_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "func_name" \
  --output_dir "codet5_func_python-150_prefix" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 32 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "f1_subtoken" \
  --prefix_tuning

python generate_predictions.py \
  --checkpoint "codet5_func_python-150_prefix/best_checkpoint" \
  --base_model "Salesforce/codet5-base" \
  --tokenizer_source "Salesforce/codet5-base" \
  --tokenizer_target "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_func_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "func_name" \
  --max_length_source 256 \
  --max_length_target 32 \
  --prefix_tuning

