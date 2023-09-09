import os.path

import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from args import DataArguments, EvaluationArguments
from train import tokenize_function
from utils import load_splits, save_list


def main():
    parser = HfArgumentParser((DataArguments, EvaluationArguments))
    data_args, eval_args = parser.parse_args_into_dataclasses()

    model = AutoModelForSeq2SeqLM.from_pretrained(eval_args.checkpoint).cuda()
    tokenizer_source = AutoTokenizer.from_pretrained(eval_args.tokenizer_source)
    tokenizer_target = AutoTokenizer.from_pretrained(eval_args.tokenizer_target)

    dataset = load_splits(data_args)["test"]
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
    generation_config = GenerationConfig(max_length=eval_args.max_length_target,
                                         num_beams=eval_args.num_beams)
    preds = []
    gold = []
    no_dup_preds = []
    no_dup_gold = []
    dup_preds = []
    dup_gold = []
    for i in tqdm(range(len(dataset)), desc='Pred loop'):
        is_duplicated = dataset[i]['is_duplicated']
        attn = torch.tensor([dataset[i]['attention_mask']]).cuda()
        ids = torch.tensor([dataset[i]['input_ids']]).cuda()
        summary_ids = model.generate(input_ids=ids, attention_mask=attn,
                                     generation_config=generation_config)
        pred = tokenizer_target.decode(summary_ids[0], skip_special_tokens=True)
        truth = tokenizer_target.decode([l for l in dataset[i]["labels"] if l != -100], skip_special_tokens=True)
        # check the decoding
        print(f'Pred: {pred.strip()} -- Truth: {truth.strip()}')
        preds.append(pred.strip())
        gold.append(truth.strip())
        if not is_duplicated:
            no_dup_gold.append(truth.strip())
            no_dup_preds.append(pred.strip())
        else:
            dup_gold.append(truth.strip())
            dup_preds.append(pred.strip())

    save_list(preds, os.path.join(eval_args.checkpoint, 'predictions_full.txt'), eval_args.include_idx)
    save_list(gold, os.path.join(eval_args.checkpoint, 'references_full.txt'), eval_args.include_idx)

    save_list(no_dup_preds, os.path.join(eval_args.checkpoint, 'predictions_no_dup.txt'), eval_args.include_idx)
    save_list(no_dup_gold, os.path.join(eval_args.checkpoint, 'references_no_dup.txt'), eval_args.include_idx)

    save_list(dup_preds, os.path.join(eval_args.checkpoint, 'predictions_dup.txt'), eval_args.include_idx)
    save_list(dup_gold, os.path.join(eval_args.checkpoint, 'references_dup.txt'), eval_args.include_idx)


if __name__ == '__main__':
    main()
