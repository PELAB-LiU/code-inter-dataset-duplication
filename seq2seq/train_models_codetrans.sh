# codeT5 lora
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --output_dir "codet5_java2csharp_peft" \
  --max_length_source 512 \
  --max_length_target 512 \
  --num_train_epochs 150 \
  --patience 3 \
  --generation_max_length 512 \
  --metric_for_best_model "bleu-codetrans-cxg" \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --eval_steps 5000 \
  --max_steps 100000 \
  --per_device_train_batch_size 16 \
  --save_steps 5000 \
  --peft

python generate_predictions.py \
  --checkpoint "codet5_java2csharp_peft/best_checkpoint" \
  --base_model "Salesforce/codet5-base" \
  --tokenizer_source "Salesforce/codet5-base" \
  --tokenizer_target "Salesforce/codet5-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512 \
  --peft



# codeT5
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "Salesforce/codet5-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --output_dir "codet5_java2csharp" \
  --max_length_source 512 \
  --max_length_target 512 \
  --num_train_epochs 150 \
  --patience 3 \
  --generation_max_length 512 \
  --metric_for_best_model "bleu-codetrans-cxg" \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --eval_steps 5000 \
  --max_steps 100000 \
  --per_device_train_batch_size 16 \
  --save_steps 5000

python generate_predictions.py \
  --checkpoint "codet5_java2csharp/best_checkpoint" \
  --tokenizer_source "Salesforce/codet5-base" \
  --tokenizer_target "Salesforce/codet5-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512


#T5
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "t5-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --output_dir "t5_java2csharp" \
  --max_length_source 512 \
  --max_length_target 512 \
  --num_train_epochs 150 \
  --patience 3 \
  --generation_max_length 512 \
  --metric_for_best_model "bleu-codetrans-cxg" \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --eval_steps 5000 \
  --max_steps 100000 \
  --per_device_train_batch_size 16 \
  --save_steps 5000


python generate_predictions.py \
  --checkpoint "t5_java2csharp/best_checkpoint" \
  --tokenizer_source "t5-base" \
  --tokenizer_target "t5-base" \
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
  --output_dir "codet5_java2csharp_prefix" \
  --max_length_source 512 \
  --max_length_target 512 \
  --num_train_epochs 150 \
  --patience 3 \
  --generation_max_length 512 \
  --metric_for_best_model "bleu-codetrans-cxg" \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --eval_steps 5000 \
  --max_steps 100000 \
  --per_device_train_batch_size 16 \
  --save_steps 5000 \
  --prefix_tuning

python generate_predictions.py \
  --checkpoint "codet5_java2csharp_prefix/best_checkpoint" \
  --base_model "Salesforce/codet5-base" \
  --tokenizer_source "Salesforce/codet5-base" \
  --tokenizer_target "Salesforce/codet5-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512 \
  --prefix_tuning


# bart
python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "facebook/bart-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --output_dir "bart_java2csharp" \
  --max_length_source 512 \
  --max_length_target 512 \
  --num_train_epochs 150 \
  --patience 3 \
  --generation_max_length 512 \
  --metric_for_best_model "bleu-codetrans-cxg" \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --eval_steps 5000 \
  --max_steps 100000 \
  --per_device_train_batch_size 16 \
  --save_steps 5000

python generate_predictions.py \
  --checkpoint "bart_java2csharp/best_checkpoint" \
  --tokenizer_source "facebook/bart-base" \
  --tokenizer_target "facebook/bart-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512

# rand

python train.py \
  --architecture "encoder-decoder" \
  --encoder_decoder "control_models/model_0" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --output_dir "rand0_java2csharp" \
  --max_length_source 512 \
  --max_length_target 512 \
  --num_train_epochs 150 \
  --patience 3 \
  --generation_max_length 512 \
  --metric_for_best_model "bleu-codetrans-cxg" \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --eval_steps 5000 \
  --max_steps 100000 \
  --per_device_train_batch_size 16 \
  --save_steps 5000

python generate_predictions.py \
  --checkpoint "rand0_java2csharp/best_checkpoint" \
  --tokenizer_source "control_models/model_0" \
  --tokenizer_target "control_models/model_0" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512



python train.py \
  --architecture "rand+rand" \
  --encoder "microsoft/codebert-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --output_dir "rand_java2csharp" \
  --max_length_source 512 \
  --max_length_target 512 \
  --num_train_epochs 150 \
  --patience 3 \
  --generation_max_length 512 \
  --metric_for_best_model "bleu-codetrans-cxg" \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --eval_steps 5000 \
  --max_steps 100000 \
  --per_device_train_batch_size 16 \
  --save_steps 5000

python generate_predictions.py \
  --checkpoint "rand_java2csharp/best_checkpoint" \
  --tokenizer_source "microsoft/codebert-base" \
  --tokenizer_target "microsoft/codebert-base" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512


python train.py \
  --architecture "encoder+rand" \
  --encoder "dbernsohn/roberta-python" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --output_dir "robertapython_java2csharp" \
  --max_length_source 512 \
  --max_length_target 512 \
  --num_train_epochs 150 \
  --patience 3 \
  --generation_max_length 512 \
  --metric_for_best_model "bleu-codetrans-cxg" \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --eval_steps 5000 \
  --max_steps 100000 \
  --per_device_train_batch_size 16 \
  --save_steps 5000

python generate_predictions.py \
  --checkpoint "robertapython_java2csharp/best_checkpoint" \
  --tokenizer_source "dbernsohn/roberta-python" \
  --tokenizer_target "dbernsohn/roberta-python" \
  --data_path_hf "antolin/codetrans_interduplication" \
  --source_column "snippet" \
  --target_column "cs" \
  --max_length_source 512 \
  --max_length_target 512