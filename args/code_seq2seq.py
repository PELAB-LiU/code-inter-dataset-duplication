from dataclasses import field, dataclass
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Salesforce/codet5-small")


@dataclass
class DataArguments:
    data_path: str = field(default='python-150/data_parsed_nl.jsonl', metadata={"help": "Path to the dataset."})
    interduplicates_path: str = field(default='python-150/interduplicates.json',
                                      metadata={"help": "Path to the dataset."})
    representatives_path: str = field(default='python-150/representatives.json',
                                      metadata={"help": "Path to the representatives."})
    source_column: str = field(default="snippet", metadata={"help": "Name of the input column."})
    target_column: str = field(default="nl_parsed", metadata={"help": "Name of the output column."})
    test_size: float = field(default=0.3)
    prefix: str = field(default="Summarize Python: ")


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    max_length: int = field(default=256)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    fp16: bool = field(default=True)
    save_strategy: str = field(default="epoch")
    logging_steps: int = field(default=100)
    learning_rate: float = field(default=5e-5)
    seed: int = field(default=123)
    max_grad_norm: float = field(default=1.)
    output_dir: str = field(default='python-150_codet5')
    evaluation_strategy: str = field(default='epoch')
    predict_with_generate: bool = field(default=True)
    generation_num_beams: int = field(default=10)
    generation_max_length: int = field(default=256)

