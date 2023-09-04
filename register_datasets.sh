

for DATASET in codesearchnet bigclonebench codetrans fcm python-150 tlc ; do
  python register_in_db.py --data ${DATASET}/data.jsonl --meta ${DATASET}/metadata.yaml
done