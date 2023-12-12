import os

import numpy as np
from transformers import HfArgumentParser, Seq2SeqTrainer, EarlyStoppingCallback

from args import ModelArguments, TrainingArguments, DataArguments
from evaluation_metrics import get_normalization, f1_subtokens, nltk_sentence_bleu
from utils import load_splits, load_model_tokenizers_seq2seq, set_seed


def tokenize_function(examples, prefix, tokenizer_source, tokenizer_target, source_column,
                      target_column, max_length_source, max_length_target,
                      is_split_source, is_split_target):
    prefix = '' if not prefix else prefix

    if is_split_source:
        inputs = [prefix + ' '.join(example) for example in examples[source_column]]
    else:
        inputs = [prefix + example for example in examples[source_column]]
    if is_split_target:
        targets = [' '.join(example) for example in examples[target_column]]
    else:
        targets = [example for example in examples[target_column]]
    model_inputs = tokenizer_source(inputs, max_length=max_length_source, padding="max_length", truncation=True)
    labels = tokenizer_target(targets, max_length=max_length_target, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"].copy()
    model_inputs["labels"] = [
        [(l if l != tokenizer_target.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
    ]
    return model_inputs


def f1_subtokens_python(pred, label):
    pred = [p.lower() for p in pred.split('_') if p != '']
    label = [l.lower() for l in label.split('_') if l != '']
    if len(pred) == 0:
        return 0.
    prec = len([p for p in pred if p in label]) / len(pred)
    recall = len([l for l in label if l in pred]) / len(label)
    if prec + recall == 0:
        return 0.
    else:
        return 2 * prec * recall / (prec + recall)


def compute_metrics(eval_pred, tokenizer, task="code2text"):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    normalization = get_normalization(task)
    decoded_preds = [normalization(pred)
                     for pred in decoded_preds]
    decoded_labels = [normalization(label)
                      for label in decoded_labels]
    assert len(decoded_labels) == len(decoded_preds)
    if task == "func":
        f1s = [f1_subtokens(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
        return {"func": np.round(np.mean(f1s), 4)}
    else:
        bleu_full = [nltk_sentence_bleu(p, r) for p, r in zip(decoded_preds, decoded_labels)]
        return {task: np.round(np.mean(bleu_full), 4)}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    model, tokenizer_source, tokenizer_target = load_model_tokenizers_seq2seq(model_args)

    dataset = load_splits(data_args)
    dataset = dataset.map(lambda examples: tokenize_function(examples,
                                                             prefix=data_args.prefix,
                                                             tokenizer_source=tokenizer_source,
                                                             tokenizer_target=tokenizer_target,
                                                             is_split_source=data_args.is_split_source,
                                                             is_split_target=data_args.is_split_target,
                                                             max_length_source=TrainingArguments.max_length_source,
                                                             max_length_target=TrainingArguments.max_length_target,
                                                             source_column=data_args.source_column,
                                                             target_column=data_args.target_column),
                          batched=True, load_from_cache_file=False, num_proc=8).remove_columns([data_args.source_column,
                                                                                                data_args.target_column])
    callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.patience)]

    # prefix tuning has a bug when load_best_model_at_end is true and projection is true.
    # this is why I do this
    # at some point I should report the bug in HF
    # it is not critical as the best model is normally the last one
    if model_args.prefix_tuning:
        training_args.load_best_model_at_end = False
        callbacks = []
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        callbacks=callbacks,
        # compute_metrics=lambda x: compute_metrics(x, tokenizer_target, training_args.metric_for_best_model)
    )
    trainer.train()

    trainer.save_model(os.path.join(training_args.output_dir, 'best_checkpoint'))


if __name__ == '__main__':
    main()
