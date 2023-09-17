from dataclasses import field, dataclass
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    encoder: Optional[str] = field(default="microsoft/codebert-base")
    is_baseline: bool = field(default=False)


@dataclass
class DataArguments:
    data_path_hf: Optional[str] = field(default="antolin/bigclonebench_interduplication",
                                        metadata={"help": "Path to the hf dataset."})
    tokens_1: Optional[str] = field(default="tokens1", metadata={"help": "Name of the input column."})
    tokens_2: Optional[str] = field(default="tokens2", metadata={"help": "Name of the output column."})
    label: Optional[str] = field(default="label", metadata={"help": "Name of the output column."})


@dataclass
class TrainingArguments(TrainingArguments):
    do_train: bool = field(default=False)
    max_length: int = field(default=512)
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=16)
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
    load_best_model_at_end: bool = field(default=True)
    save_total_limit: int = field(default=1)

    metric_for_best_model: str = field(default="f1")
    greater_is_better: bool = field(default=True)
