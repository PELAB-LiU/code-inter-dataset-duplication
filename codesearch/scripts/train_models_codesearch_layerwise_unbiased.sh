seeds=(1 2 3 4 5)
t=(1 2 3 4 5 6 7 8 9 10 11)

model_name="csn-small-unbiased-random-20"
hf_model_name="antolin/$model_name"

for seed in "${seeds[@]}";
do
echo "Running seed $seed"
    for t in "${t[@]}";
    do

    python train.py \
    --checkpoint "results/layerwise/$seed/$model_name-$t.bin" \
    --data_path_hf "antolin/python-150_interduplication" \
    --tokens_column "tokens" \
    --nl_column "nl" \
    --num_train_epochs 10 \
    --max_code_len 256 \
    --max_nl_len 128 \
    --do_train \
    --telly $t \
    --model_name_or_path "$hf_model_name" \
    --seed $seed
    done
done