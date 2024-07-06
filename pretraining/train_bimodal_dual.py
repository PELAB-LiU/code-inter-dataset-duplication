from argparse import ArgumentParser

from datasets import load_dataset, concatenate_datasets
from transformers import Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, \
    T5ForConditionalGeneration, Seq2SeqTrainingArguments


def tokenize_function(examples, tokenizer, what_input='snippets', max_length=512):
    snippets = [' '.join(tokens) for tokens in examples["tokens"]]
    nls = [nl for nl in examples["nl"]]

    if what_input == 'snippets':
        inputs = snippets
        targets = nls
    else:
        inputs = nls
        targets = snippets

    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"].copy()
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
    ]
    return model_inputs


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    config = AutoConfig.from_pretrained(args.base_model)
    config.num_layers = args.num_layers
    config.num_decoder_layers = args.num_decoder_layers
    print(config)
    model = T5ForConditionalGeneration(config)

    dataset = load_dataset(args.dataset)[args.split]
    dataset_snippets = dataset.map(lambda examples: tokenize_function(examples,
                                                                      tokenizer=tokenizer,
                                                                      what_input='snippets',
                                                                      max_length=args.block_size),
                                   batched=True, load_from_cache_file=False, num_proc=8).remove_columns(
        dataset.column_names)
    dataset_nls = dataset.map(lambda examples: tokenize_function(examples,
                                                                 tokenizer=tokenizer,
                                                                 what_input='nls',
                                                                 max_length=args.block_size),
                              batched=True, load_from_cache_file=False, num_proc=8).remove_columns(
        dataset.column_names)
    dataset = concatenate_datasets([dataset_snippets, dataset_nls])

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.checkpoint,
        learning_rate=5e-5,
        num_train_epochs=20,
        push_to_hub=False,
        logging_strategy="steps",
        save_strategy="epoch",
        logging_steps=100,
        max_grad_norm=1,
        load_best_model_at_end=False,
        seed=123,
        evaluation_strategy="no",
        per_device_train_batch_size=args.batch_size
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for training mlm')
    parser.add_argument('--dataset', choices=['antolin/csn-small-interduplication', 'antolin/csn-interduplication'],
                        default='antolin/csn-small-interduplication')
    parser.add_argument('--split', choices=['biased', 'unbiased'], default='biased')
    parser.add_argument('--checkpoint', default='/data/ja_models/pre-trained/csn_small/random-encoder-decoder-biased')
    parser.add_argument('--base_model', default='Salesforce/codet5-base')
    parser.add_argument('--block_size', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_decoder_layers', default=12, type=int)
    parser.add_argument('--num_layers', default=12, type=int)
    args = parser.parse_args()
    main(args)
