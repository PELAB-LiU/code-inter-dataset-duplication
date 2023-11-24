from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('antolin/python-150_interduplication')['test']

dataset = dataset.map(lambda examples: tokenize_function(examples,
                                                             prefix=data_args.prefix,
                                                             tokenizer_source=tokenizer_source,
                                                             tokenizer_target=tokenizer_target,
                                                             is_split_source=data_args.is_split_source,
                                                             is_split_target=data_args.is_split_target,
                                                             max_length_source=eval_args.max_length_source,
                                                             max_length_target=eval_args.max_length_target,
                                                             source_column=data_args.source_column,
                                                             target_column=data_args.target_column),
                          batched=True, load_from_cache_file=False, num_proc=8).remove_columns([data_args.source_column,
                                                                                                data_args.target_column])
print(dataset.filter(lambda x: x['nl'].strip() == ''))

l = ''
for i in tqdm(range(len(dataset)), desc='Pred loop'):
    l.append()