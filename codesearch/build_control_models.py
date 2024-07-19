import argparse

from train import set_seed
from transformers import AutoConfig, AutoTokenizer, RobertaModel


def main(args):
    # just roberta models
    config = AutoConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # set seed
    set_seed(args.seed)

    for i in range(args.number_control_models):
        model = RobertaModel(config)
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
