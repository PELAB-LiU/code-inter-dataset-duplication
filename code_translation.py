from transformers import HfArgumentParser, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer

from args.code_translation_args import ModelArguments, TrainingArguments, DataArguments
from code_search import set_seed, load_splits, COLUMN_INTER_DUPLICATED


def tokenize_function(examples, prefix, tokenizer, source_column, target_column, max_length):
    inputs = [prefix + example for example in examples[source_column]]
    targets = [example for example in examples[target_column]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    dataset = load_splits(data_args.data_path, data_args.interduplicates_path, data_args.test_size, training_args.seed)
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in
                                      [data_args.tokens_column, data_args.nl_column, COLUMN_INTER_DUPLICATED]])
    dataset = dataset.map(lambda example: {data_args.tokens_column: ' '.join(example[data_args.tokens_column])})
    dataset = dataset.map(lambda examples: tokenize_function(examples,
                                                             prefix=data_args.prefix,
                                                             tokenizer=tokenizer,
                                                             max_length=TrainingArguments.max_length,
                                                             source_column=data_args.tokens_column,
                                                             target_column=data_args.nl_column),
                          batched=True, load_from_cache_file=True).remove_columns([data_args.tokens_column,
                                                                                   data_args.nl_column])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_args.model_name_or_path)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

    print('hi!')
    print(model)


if __name__ == '__main__':
    main()
