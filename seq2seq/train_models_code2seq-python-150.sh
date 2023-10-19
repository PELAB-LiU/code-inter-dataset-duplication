# codeT5 lora
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "codet5_code2text_python-150_lora" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128 \
  --metric_for_best_model "code2text" \
  --lora \
  --learning_rate 3e-4

python generate_predictions.py \
  --checkpoint "codet5_code2text_python-150_lora/best_checkpoint" \
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
  --generation_max_length 128

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

# codeT5 prefix tuning
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "codet5_code2text_python-150_prefix" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 32 \
  --patience 3 \
  --generation_max_length 128 \
  --prefix_tuning \
  --learning_rate 3e-4

python generate_predictions.py \
  --checkpoint "codet5_code2text_python-150_prefix/best_checkpoint" \
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

# rand rand
python train.py \
  --architecture "rand+rand" \
  --encoder "microsoft/codebert-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "rand_code2text_python-150" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128

python generate_predictions.py \
  --checkpoint "rand_code2text_python-150/best_checkpoint" \
  --tokenizer_source "microsoft/codebert-base" \
  --tokenizer_target "microsoft/codebert-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128

# roberta java
python train.py \
  --architecture "encoder+rand" \
  --encoder "dbernsohn/roberta-java" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "robertajava_code2text_python-150" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128

python generate_predictions.py \
  --checkpoint "robertajava_code2text_python-150/best_checkpoint" \
  --tokenizer_source "dbernsohn/roberta-java" \
  --tokenizer_target "dbernsohn/roberta-java" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128


# roberta javascript
python train.py \
  --architecture "encoder+rand" \
  --encoder "dbernsohn/roberta-javascript" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "robertajavascript_code2text_python-150" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128

python generate_predictions.py \
  --checkpoint "robertajavascript_code2text_python-150/best_checkpoint" \
  --tokenizer_source "dbernsohn/roberta-javascript" \
  --tokenizer_target "dbernsohn/roberta-javascript" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128

# bart
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "facebook/bart-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "bart_code2text_python-150" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 128 \
  --patience 3 \
  --generation_max_length 128

python generate_predictions.py \
  --checkpoint "bart_code2text_python-150/best_checkpoint" \
  --tokenizer_source "facebook/bart-base" \
  --tokenizer_target "facebook/bart-base" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128

# codet5 large
# codeT5 prefix
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-large" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "codet5_large_code2text_python-150_prefix" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 32 \
  --patience 3 \
  --generation_max_length 128 \
  --prefix_tuning \
  --learning_rate 3e-4

python generate_predictions.py \
  --checkpoint "codet5_large_code2text_python-150_prefix/best_checkpoint" \
  --base_model "Salesforce/codet5-large" \
  --tokenizer_source "Salesforce/codet5-large" \
  --tokenizer_target "Salesforce/codet5-large" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128 \
  --prefix_tuning

# codet5 small
# codeT5 prefix
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-small" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --output_dir "codet5_small_code2text_python-150_prefix" \
  --num_train_epochs 10 \
  --max_length_source 256 \
  --max_length_target 32 \
  --patience 3 \
  --generation_max_length 128 \
  --prefix_tuning \
  --learning_rate 3e-4

python generate_predictions.py \
  --checkpoint "codet5_small_code2text_python-150_prefix/best_checkpoint" \
  --base_model "Salesforce/codet5-small" \
  --tokenizer_source "Salesforce/codet5-small" \
  --tokenizer_target "Salesforce/codet5-small" \
  --data_path_hf "antolin/python-150_interduplication" \
  --source_column "tokens" \
  --is_split_source \
  --target_column "nl" \
  --max_length_source 256 \
  --max_length_target 128 \
  --prefix_tuning