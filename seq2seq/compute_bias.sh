model="codet5large_lora"
folder="/data/ja_models/final_models/codetrans"
lang="java"
task="codetrans"
dataset="antolin/codetrans_interduplication"

python get_csv_model.py --folder $folder --model $model --lang $lang --task $task --dataset $dataset --output model.csv

control_models=("bart" "rand66" "rand33" "rand63" "t5v1" "t5_fpfalse")

for control_model in "${control_models[@]}";
do
  echo "Control model: $control_model"
  python get_csv_model.py --folder $folder --model $control_model --lang $lang --task $task --dataset $dataset --output control_model.csv
  python importance_sampling.py --csv control_model.csv --output judgement.json
  python get_bias.py --csv model.csv --judgement judgement.json
  echo "---------------------------------------"
done
