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
