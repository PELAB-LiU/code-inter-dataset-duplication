
for t in {12..1}
do
  python train.py \
    --checkpoint "graphcodebert_python-150_telly$t.bin" \
    --data_path_hf "antolin/python-150_interduplication" \
    --tokens_column "tokens" \
    --nl_column "nl" \
    --num_train_epochs 5 \
    --max_code_len 256 \
    --max_nl_len 128 \
    --do_train \
    --telly $t \
    --model_name_or_path "microsoft/graphcodebert-base"
done