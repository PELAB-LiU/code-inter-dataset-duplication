
# for each dataset
for DATASET in codesearchnet bigclonebench code-trans java-size javaCorpus ; do
  # python register_in_db.py --data dataset_name/data.jsonl --meta dataset_name/metadata.yaml
  python register_in_db.py --data ${DATASET}/data.jsonl --meta ${DATASET}/metadata.yaml
done

python compute_duplicates.py --lang java