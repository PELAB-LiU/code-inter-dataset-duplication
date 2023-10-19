from dataclasses import field, dataclass
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    architecture: Optional[str] = field(default=None,
                                        metadata={"help": "encoder+rand, encoder+decoder, shared, encoder-decoder"})
    encoder_decoder: Optional[str] = field(default=None)
    encoder: Optional[str] = field(default="microsoft/codebert-base")
    decoder: Optional[str] = field(default="microsoft/codebert-base")
    decoder_rand_layers: int = field(default=6)
    telly: bool = field(default=False)
    lora: bool = field(default=False)
    prefix_tuning: bool = field(default=False)
    r: int = field(default=8)
    alpha: int = field(default=16)


@dataclass
class DataArguments:
    data_path_train: Optional[str] = field(default=None, metadata={"help": "Path to the train dataset."})
    data_path_test: Optional[str] = field(default=None, metadata={"help": "Path to the test dataset."})
    data_path_dev: Optional[str] = field(default=None, metadata={"help": "Path to the dev dataset."})

    data_path_hf: Optional[str] = field(default=None,
                                        metadata={"help": "Path to the hf dataset."})

    source_column: Optional[str] = field(default=None, metadata={"help": "Name of the input column."})
    is_split_source: bool = field(default=False)
    target_column: str = field(default=None, metadata={"help": "Name of the output column."})
    is_split_target: bool = field(default=False)

    prefix: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    max_length_source: int = field(default=256)
    max_length_target: int = field(default=128)
    num_train_epochs: int = field(default=10)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    fp16: bool = field(default=True)
    save_strategy: str = field(default="epoch")
    logging_steps: int = field(default=100)
    learning_rate: float = field(default=5e-5)
    seed: int = field(default=123)
    max_grad_norm: float = field(default=1.)
    output_dir: str = field(default=None)
    evaluation_strategy: str = field(default="epoch")
    patience: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    save_total_limit: int = field(default=1)

    predict_with_generate: bool = field(default=False)
    generation_max_length: int = field(default=128)
    generation_num_beams: int = field(default=10)
    metric_for_best_model: str = field(default=None)  # code2text, func, codetrans
    greater_is_better: bool = field(default=None)


@dataclass
class EvaluationArguments:
    checkpoint: Optional[str] = field(default=None)
    tokenizer_source: Optional[str] = field(default=None)
    tokenizer_target: Optional[str] = field(default=None)
    max_length_source: int = field(default=256)
    max_length_target: int = field(default=128)
    num_beams: int = field(default=10)
    include_idx: bool = field(default=False)
    lora: bool = field(default=False)
    base_model: str = field(default=None)
    prefix_tuning: bool = field(default=False)
