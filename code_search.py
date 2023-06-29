import json
import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import ttest_ind
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import retrieval_reciprocal_rank
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from args.code_search_args import ModelArguments, DataArguments, TrainingArguments

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DualEncoderModel(nn.Module):
    def __init__(self, code_encoder, nl_encoder):
        super().__init__()
        self.code_encoder = code_encoder
        self.nl_encoder = nl_encoder

    def forward(self, input_ids_code, inputs_ids_nl,
                attention_mask_code, attention_mask_nl):
        emb_code = self.code_encoder(input_ids_code, attention_mask_code)
        emb_nl = self.nl_encoder(inputs_ids_nl, attention_mask_nl)
        return emb_code['pooler_output'], emb_nl['pooler_output']


COLUMN_INTER_DUPLICATED = "is_inter_duplicated"


def load_splits(data_path, interduplicates_path, test_size, seed):
    dataset = load_dataset('json', data_files=data_path)
    with open(interduplicates_path) as f:
        interduplicates = set(json.load(f))
    dataset = dataset.map(lambda example: {
        COLUMN_INTER_DUPLICATED: example["id_within_dataset"] in interduplicates
    }, remove_columns="id_within_dataset")
    dataset = dataset.class_encode_column(COLUMN_INTER_DUPLICATED)
    return dataset["train"].train_test_split(test_size=test_size,
                                             stratify_by_column=COLUMN_INTER_DUPLICATED,
                                             seed=seed)


def tokenize_function(examples, tokenizer, max_len, column):
    dic = tokenizer(examples[column], padding="max_length", truncation=True, max_length=max_len)
    dic_new = {x + "_" + column: y for x, y in dic.items()}
    return dic_new


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# TODO set input to train
def train(train_set, model, checkpoint, batch_size_train=32, lr=5e-5, epochs=1, gradient_accumulation=1,
          max_grad_norm=1,
          log_steps=100, input_ids_code='input_ids_tokens', inputs_ids_nl='input_ids_nl',
          attention_mask_code='attention_mask_tokens', attention_mask_nl='attention_mask_nl'):
    train_set.set_format("torch")
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size_train)
    model.to(DEVICE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, eps=1e-8)

    criterion = torch.nn.CrossEntropyLoss()

    logger.info('Training phase!')
    logger.info(f'Effective batch size: {batch_size_train * gradient_accumulation}')
    logger.info(f'Initial lr: {lr}')
    logger.info(f'Epochs: {epochs}')
    logger.info(f'Parameters: {sum(map(torch.numel, filter(lambda p: p.requires_grad, model.parameters())))}')

    num_training_steps = epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps * 0.1,
                                                num_training_steps=num_training_steps)
    progress_bar = tqdm(range(num_training_steps))
    steps = 0
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        model.train()
        for j, batch in enumerate(train_dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            emb_code, emb_nl = model(input_ids_code=batch[input_ids_code], inputs_ids_nl=batch[inputs_ids_nl],
                                     attention_mask_code=batch[attention_mask_code],
                                     attention_mask_nl=batch[attention_mask_nl])
            scores = torch.matmul(emb_nl, torch.transpose(emb_code, 0, 1))
            loss = criterion(scores, torch.arange(scores.shape[0]).to(DEVICE))

            train_loss += loss.item()
            loss = loss / gradient_accumulation

            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)

            if ((j + 1) % gradient_accumulation == 0) or (j + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            progress_bar.update(1)
            steps += 1
            if steps % log_steps == 0:
                logger.info(
                    f'Epoch {epoch} | step={steps} | train_loss={train_loss / (j + 1):.4f}'
                )

        logger.info(
            f'Epoch {epoch} | train_loss={train_loss / len(train_dataloader):.4f}'
        )
    logger.info('Saving model!')
    torch.save(model.state_dict(), checkpoint)
    logger.info(f'Model saved: {checkpoint}')


def evaluate(eval_dataset, model, batch_size_eval=1000, input_ids_code='input_ids_tokens', inputs_ids_nl='input_ids_nl',
             attention_mask_code='attention_mask_tokens', attention_mask_nl='attention_mask_nl'):
    model.eval()
    rrs = defaultdict(list)
    eval_dataset.set_format("torch")
    dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size_eval)
    for batch in tqdm(dataloader, desc='Evaluation loop'):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            emb_code, emb_nl = model(input_ids_code=batch[input_ids_code], inputs_ids_nl=batch[inputs_ids_nl],
                                     attention_mask_code=batch[attention_mask_code],
                                     attention_mask_nl=batch[attention_mask_nl])
            scores = torch.matmul(emb_nl, torch.transpose(emb_code, 0, 1))
        for scs, tgt, group in zip(scores, torch.eye(scores.shape[0]).to(DEVICE), batch[COLUMN_INTER_DUPLICATED]):
            rr = retrieval_reciprocal_rank(scs, tgt).item()
            rrs[int(group.item())].append(rr)
    return rrs


def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    model = AutoModel.from_pretrained(model_args.model_name_or_path)
    dual_encoder_model = DualEncoderModel(model, model)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    dataset = load_splits(data_args.data_path, data_args.interduplicates_path, data_args.test_size, training_args.seed)
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in
                                      [data_args.tokens_column, data_args.nl_column, COLUMN_INTER_DUPLICATED]])
    dataset = dataset.map(lambda example: {data_args.tokens_column: ' '.join(example[data_args.tokens_column])})
    # print(Dataset.to_pandas(dataset["train"]).describe())
    # print(Dataset.to_pandas(dataset["test"]).describe())

    dataset = dataset.map(lambda examples: tokenize_function(examples,
                                                             tokenizer=tokenizer,
                                                             max_len=TrainingArguments.max_code_len,
                                                             column=data_args.tokens_column),
                          batched=True, load_from_cache_file=True).remove_columns([data_args.tokens_column])
    dataset = dataset.map(lambda examples: tokenize_function(examples,
                                                             tokenizer=tokenizer,
                                                             max_len=TrainingArguments.max_nl_len,
                                                             column=data_args.nl_column),
                          batched=True, load_from_cache_file=True).remove_columns([data_args.nl_column])
    full_test_dataset = dataset["test"]
    # without_interdup = full_test_dataset.filter(lambda example: example[COLUMN_INTER_DUPLICATED] <= 0)
    # just_interdup = full_test_dataset.filter(lambda example: example["is_inter_duplicated"] > 0)

    train(train_set=dataset["train"],
          model=dual_encoder_model,
          checkpoint=model_args.checkpoint,
          batch_size_train=training_args.batch_size_eval,
          lr=training_args.learning_rate,
          epochs=training_args.num_train_epochs,
          gradient_accumulation=training_args.gradient_accumulation_steps,
          max_grad_norm=training_args.max_grad_norm,
          log_steps=training_args.logging_steps,
          input_ids_code='input_ids_tokens',
          inputs_ids_nl='input_ids_nl',
          attention_mask_code='attention_mask_tokens',
          attention_mask_nl='attention_mask_nl')
    rrs = evaluate(eval_dataset=full_test_dataset,
                   model=dual_encoder_model,
                   batch_size_eval=training_args.batch_size_eval,
                   input_ids_code='input_ids_tokens',
                   inputs_ids_nl='input_ids_nl',
                   attention_mask_code='attention_mask_tokens',
                   attention_mask_nl='attention_mask_nl')
    mrrs = {x: np.mean(y) for x, y in rrs.items()}
    logger.info(f'MRRs: {mrrs}')
    logger.info(f'T-test: {ttest_ind(rrs[0], rrs[1]).pvalue:.4f}')
    logger.info(f'Cohen d: {cohend(rrs[0], rrs[1]):.4f}')


if __name__ == '__main__':
    main()
