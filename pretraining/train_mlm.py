from argparse import ArgumentParser

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoConfig, RobertaForMaskedLM


def main(args):
    print(f'Base model {args.base_model}')
    print(f'Split {args.split}')
    print(f'Checkpoint {args.checkpoint}')
    print(f'Dataset {args.dataset}')

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if not args.initialize_random:
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
    else:
        config = AutoConfig.from_pretrained(args.base_model)
        config.num_hidden_layers = args.layers
        model = RobertaForMaskedLM(config)
    dataset = load_dataset(args.dataset)[args.split] #.select(range(0, 2000))

    def tokenize_function(examples):
        # codes = [' '.join(example) for example in examples["tokens"]]
        # nls = [' '.join(example.strip().split()) for example in examples["nl"]]
        # pairs = [[c, nl] for c, nl in zip(codes, nls)]
        # print(pairs[0])
        # def max ( a , b ) :
        # def max(a, b):
        pairs = [[' '.join(snippet.strip().split()), ' '.join(nl.strip().split())]
                 for snippet, nl in zip(examples["snippet"], examples["nl"])]
        result = tokenizer(pairs, max_length=args.block_size, padding="max_length", truncation=True)
        result["labels"] = result["input_ids"].copy()
        # print(result["labels"][0])
        # print(tokenizer.decode(result["labels"][0]))
        return result

    dataset = dataset.map(tokenize_function, batched=True, num_proc=12, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
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
        # weight_decay=0.01,
        evaluation_strategy="no",
        per_device_train_batch_size=args.batch_size
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for training mlm')
    parser.add_argument('--dataset', choices=['antolin/csn-small-interduplication', 'antolin/csn-interduplication'],
                        default='antolin/csn-small-interduplication')
    parser.add_argument('--split', choices=['biased', 'unbiased'], default='biased')
    parser.add_argument('--checkpoint', default='/data/ja_models/pre-trained/csn_big/roberta-biased')
    parser.add_argument('--base_model', default='roberta-base')
    parser.add_argument('--initialize_random', action='store_true')
    parser.add_argument('--block_size', default=512, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--layers', default=12, type=int)
    args = parser.parse_args()
    main(args)
