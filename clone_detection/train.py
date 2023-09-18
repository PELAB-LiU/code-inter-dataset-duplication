import os

import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import HfArgumentParser, RobertaForSequenceClassification, AutoTokenizer, Trainer

from args import ModelArguments, DataArguments, TrainingArguments


def tokenize_data(examples, tokenizer, max_len, tokens_1, tokens_2):
    c1s = [' '.join(example) for example in examples[tokens_1]]
    c2s = [' '.join(example) for example in examples[tokens_2]]
    pairs = [[c, nl] for c, nl in zip(c1s, c2s)]
    return tokenizer(pairs, max_length=max_len, padding="max_length", truncation=True)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_train:
        model = RobertaForSequenceClassification.from_pretrained(model_args.encoder, num_labels=2)
        if model_args.is_baseline:
            config = model.config
            config.num_hidden_layers = 6
            model = RobertaForSequenceClassification(config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(os.path.join(training_args.output_dir,
                                                                              'best_checkpoint'),
                                                                 num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_args.encoder)

    dataset = load_dataset(data_args.data_path_hf)
    dataset.rename_column(data_args.label, 'labels')

    dataset['train'] = dataset['train'].shuffle(seed=training_args.seed).select(range(int(0.1*len(dataset['train']))))
    dataset['valid'] = dataset['valid'].shuffle(seed=training_args.seed).select(range(int(0.1*len(dataset['valid']))))
    dataset = dataset.map(
        lambda x: tokenize_data(x, tokenizer, training_args.max_length, data_args.tokens_1, data_args.tokens_2),
        batched=True, batch_size=1000)

    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        if not training_args.do_train:
            print(f1_metric.compute(predictions=predictions, references=labels))
            new_df = pd.DataFrame()
            new_df['pred_label'] = predictions
            new_df['true_label'] = labels
            new_df['logits_0'] = logits[:, 0]
            new_df['logits_1'] = logits[:, 1]
            new_df.to_csv(os.path.join(training_args.output_dir, 'best_checkpoint', 'preds_labels_logits.csv'),
                          header=True)
        return f1_metric.compute(predictions=predictions, references=labels)

    if training_args.do_train:
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_model(os.path.join(training_args.output_dir, 'best_checkpoint'))
    else:
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=None,
            eval_dataset=dataset['test'],
            compute_metrics=compute_metrics
        )
        trainer.evaluate()


if __name__ == '__main__':
    main()
