import json
import os
import random
import tempfile

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from peft import TaskType, get_peft_model, LoraConfig, PrefixTuningConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EncoderDecoderModel, AutoConfig, RobertaModel

from args import ModelArguments, DataArguments


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def load_splits(args: DataArguments):
    d = load_dataset(args.data_path_hf)
    print(d)
    if args.filter_pretraining:
        d['train'] = d['train'].filter(lambda x: not x['is_duplicated'])
        return d
    if args.augment_duplicates:
        dups = d['train'].filter(lambda x: x['is_duplicated'])
        d['train'] = concatenate_datasets([d['train'], dups])
    print(d)
    return d


def load_model_tokenizers_seq2seq(args: ModelArguments):
    if args.architecture == 'encoder-decoder':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.encoder_decoder)
        tokenizer_source = AutoTokenizer.from_pretrained(args.encoder_decoder)
        tokenizer_target = tokenizer_source
    elif args.architecture == 'encoder+decoder':
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, args.decoder)
        tokenizer_source = AutoTokenizer.from_pretrained(args.encoder)
        tokenizer_target = AutoTokenizer.from_pretrained(args.decoder)
        model.config.decoder_start_token_id = tokenizer_target.cls_token_id
        model.config.eos_token_id = tokenizer_source.eos_token_id
        if tokenizer_target.pad_token_id != tokenizer_source.pad_token_id:
            raise NotImplementedError()
        if tokenizer_target.eos_token_id != tokenizer_source.eos_token_id:
            raise NotImplementedError()
        model.config.pad_token_id = tokenizer_target.pad_token_id
    elif args.architecture == 'encoder+rand':
        tokenizer_source = AutoTokenizer.from_pretrained(args.encoder)
        tokenizer_target = tokenizer_source
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config = AutoConfig.from_pretrained(args.encoder)
            config.num_hidden_layers = args.decoder_rand_layers
            config.is_decoder = True
            rand_decoder = RobertaModel(config)
            rand_decoder.save_pretrained(tmp_dirname)
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, tmp_dirname)
            model.decoder.embeddings = model.encoder.embeddings
        model.config.decoder_start_token_id = tokenizer_target.cls_token_id
        model.config.pad_token_id = tokenizer_target.pad_token_id
    elif args.architecture == 'shared':
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, args.encoder,
                                                                    tie_encoder_decoder=True)
        tokenizer_source = AutoTokenizer.from_pretrained(args.encoder)
        tokenizer_target = tokenizer_source
        model.config.decoder_start_token_id = tokenizer_target.cls_token_id
        model.config.pad_token_id = tokenizer_target.pad_token_id
    elif args.architecture == 'rand+rand':
        tokenizer_source = AutoTokenizer.from_pretrained(args.encoder)
        tokenizer_target = tokenizer_source
        with tempfile.TemporaryDirectory() as tmp_dirname, tempfile.TemporaryDirectory() as tmp_dirname_2:
            # decoder
            config = AutoConfig.from_pretrained(args.encoder)
            config.num_hidden_layers = args.decoder_rand_layers
            config.is_decoder = True
            rand_decoder = RobertaModel(config)
            rand_decoder.save_pretrained(tmp_dirname)

            # encoder
            config = AutoConfig.from_pretrained(args.encoder)
            rand_encoder = RobertaModel(config)
            config.num_hidden_layers = args.encoder_rand_layers
            rand_encoder.save_pretrained(tmp_dirname_2)

            model = EncoderDecoderModel.from_encoder_decoder_pretrained(tmp_dirname_2, tmp_dirname)
            model.decoder.embeddings = model.encoder.embeddings
        model.config.decoder_start_token_id = tokenizer_target.cls_token_id
        model.config.pad_token_id = tokenizer_target.pad_token_id
    else:
        raise NotImplementedError()
    if args.telly:
        for p in model.encoder.parameters():
            p.requires_grad = False
    if args.lora:
        peft_config = LoraConfig(r=args.r, lora_alpha=args.alpha,
                                 lora_dropout=0.1, task_type=TaskType.SEQ_2_SEQ_LM,
                                 target_modules=['.q', '.v', '.o', '.k'])  # r=8, check lora alpha
        model = get_peft_model(model, peft_config)
    if args.prefix_tuning:
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False,
                                         num_virtual_tokens=20, prefix_projection=True)
        model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    return model, tokenizer_source, tokenizer_target


def save_list(l, path, include_idx=False):
    l_idx = [f"{i}\t{t}" for i, t in enumerate(l)]
    if not include_idx:
        with open(path, 'w') as f:
            json.dump(l, f)
    else:
        with open(path, 'w') as file:
            file.write('\n'.join(l_idx))


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
