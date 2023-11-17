from dataclasses import field, dataclass
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/codebert-base")
    checkpoint: str = field(default=None)
    is_baseline: bool = field(default=False)
    # Only active if is_baseline is true
    num_layers: int = field(default=6)
    
    telly: int = field(default=0)
    lora: bool = field(default=False)
    prefix_tuning: bool = field(default=False)


@dataclass
class DataArguments:
    data_path_hf: str = field(default=None, metadata={"help": "HF dataset."})
    tokens_column: str = field(default="tokens", metadata={"help": "Name of the tokens column."})
    nl_column: str = field(default="nl", metadata={"help": "Name of the description column."})


@dataclass
class TrainingArguments:
    do_train: bool = field(default=False)
    max_code_len: int = field(default=256)
    max_nl_len: int = field(default=128)
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=32)
    batch_size_eval: int = field(default=1000)
    gradient_accumulation_steps: int = field(default=1)
    fp16: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=250)
    logging_steps: int = field(default=100)
    learning_rate: float = field(default=5e-5)
    seed: int = field(default=123)
    max_grad_norm: float = field(default=1.)
    patience: int = field(default=2)
