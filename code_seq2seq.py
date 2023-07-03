import evaluate
import numpy as np
from transformers import HfArgumentParser, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer

from args.code_seq2seq import ModelArguments, TrainingArguments, DataArguments
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

    dataset = load_splits(data_args.data_path, data_args.interduplicates_path, data_args.representatives_path,
                          data_args.test_size, training_args.seed)
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in
                                      [data_args.source_column, data_args.target_column, COLUMN_INTER_DUPLICATED]])
    # dataset = dataset.map(lambda example: {data_args.source_column: ' '.join(example[data_args.source_column])})
    dataset = dataset.map(lambda examples: tokenize_function(examples,
                                                             prefix=data_args.prefix,
                                                             tokenizer=tokenizer,
                                                             max_length=TrainingArguments.max_length,
                                                             source_column=data_args.source_column,
                                                             target_column=data_args.target_column),
                          batched=True, load_from_cache_file=True).remove_columns([data_args.source_column,
                                                                                   data_args.target_column])

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    metric = evaluate.load("sacrebleu")
    test_dataset = dataset["test"]
    groups = np.array(test_dataset[COLUMN_INTER_DUPLICATED])

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # all
        results_all = compute_metrics_aux(preds, labels)

        # group inter
        preds_inter, labels_inter = preds[groups == 1], labels[groups == 1]
        result_inter = compute_metrics_aux(preds_inter, labels_inter)

        # group no inter
        preds_no_inter, labels_no_inter = preds[groups == 0], labels[groups == 0]
        result_no_inter = compute_metrics_aux(preds_no_inter, labels_no_inter)

        result = {"bleu": [result_inter["bleu"], result_no_inter["bleu"], results_all["bleu"]],
                  "gen_len": [result_inter["gen_len"], result_no_inter["gen_len"], results_all["gen_len"]]}

        return result

    def compute_metrics_aux(preds, labels):
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, lowercase=True)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_args.model_name_or_path)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == '__main__':
    main()
