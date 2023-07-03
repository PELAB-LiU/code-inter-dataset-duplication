from dataclasses import field, dataclass
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/unixcoder-base")
    checkpoint: str = field(default='python-150_unixcoder.bin')


@dataclass
class DataArguments:
    data_path: str = field(default='python-150/data_parsed_nl.jsonl', metadata={"help": "Path to the dataset."})
    interduplicates_path: str = field(default='python-150/interduplicates.json',
                                      metadata={"help": "Path to the interduplicates."})
    representatives_path: str = field(default='python-150/representatives.json',
                                      metadata={"help": "Path to the representatives."})
    tokens_column: str = field(default="tokens", metadata={"help": "Name of the tokens column."})
    nl_column: str = field(default="nl_parsed", metadata={"help": "Name of the description column."})
    test_size: float = field(default=0.3)


@dataclass
class TrainingArguments:
    max_code_len: int = field(default=256)
    max_nl_len: int = field(default=128)
    num_train_epochs: int = field(default=1)
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

