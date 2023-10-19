import argparse
import os
import random

import numpy as np
import torch
from transformers import AutoConfig, RobertaModel, AutoTokenizer, T5Model


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


def main(args):
    # just codet5 models
    config = AutoConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # set seed
    set_seed(args.seed)

    for i in range(args.number_control_models):
        model = T5Model(config)
        model.save_pretrained(f'{args.output_dir}/model_{i}')
        tokenizer.save_pretrained(f'{args.output_dir}/model_{i}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--number_control_models', default=3)
    parser.add_argument('--seed', default=123)
    args = parser.parse_args()
    main(args)
