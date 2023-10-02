import os
import tempfile

import numpy as np
from transformers import HfArgumentParser, Seq2SeqTrainer, EarlyStoppingCallback

from bleu_codetrans import _bleu
from bleu_code2text import computeMaps, bleuFromMaps
from args import ModelArguments, TrainingArguments, DataArguments
from utils import load_splits, load_model_tokenizers_seq2seq, save_list


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



def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip()
                     for pred in decoded_preds]
    decoded_labels = [label.strip()
                      for label in decoded_labels]
    with tempfile.TemporaryDirectory() as tmp_dirname:
        save_list(decoded_preds, os.path.join(tmp_dirname, 'predictions.txt'))
        save_list(decoded_labels, os.path.join(tmp_dirname, 'references.txt'))
        bleu_score_code_trans = _bleu(os.path.join(tmp_dirname, 'references.txt'),
                                      os.path.join(tmp_dirname, 'predictions.txt'))

        preds_indices = [f"{i}\t{t}" for i, t in enumerate(decoded_preds)]
        save_list(decoded_labels, os.path.join(tmp_dirname, 'references_idx.txt'), True)
        (goldMap, predictionMap) = computeMaps(preds_indices,
                                               os.path.join(tmp_dirname, 'references_idx.txt'))
        bleu_score_code2text = bleuFromMaps(goldMap, predictionMap)[0]

    f1s = [f1_subtokens_python(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    return {'bleu-codetrans-cxg': round(bleu_score_code_trans, 4),
            'bleu-code2text-cxg': round(bleu_score_code2text, 4),
            'f1_subtoken': np.mean(f1s)}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.patience)],
        compute_metrics=lambda x: compute_metrics(x, tokenizer_target)
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, 'best_checkpoint'))


if __name__ == '__main__':
    main()
