import tempfile

from datasets import load_dataset
from peft import TaskType, get_peft_model, LoraConfig
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
    return load_dataset(args.data_path_hf)


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
            config.num_hidden_layers = 6
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
    if args.peft:
        peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.SEQ_2_SEQ_LM)
        model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    return model, tokenizer_source, tokenizer_target


def save_list(l, path, include_idx=False):
    l_idx = [f"{i}\t{t}" for i, t in enumerate(l)]
    if not include_idx:
        with open(path, 'w') as file:
            file.write('\n'.join(l))
    else:
        with open(path, 'w') as file:
            file.write('\n'.join(l_idx))
