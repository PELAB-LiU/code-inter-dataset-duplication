import argparse

from datasets import load_dataset

KEY_WORD = 'function'


def get_input_output(tokens, lang):
    if lang == 'python':
        position = None
        for j, t in enumerate(tokens):
            if t == "def":
                position = j + 1
                break
        return [t if j != position else KEY_WORD for j, t in enumerate(tokens)], tokens[position]
    elif lang == 'java':
        position = None
        for j, t in enumerate(tokens):
            if t == '(':
                position = j - 1
                break
        return [t if j != position else KEY_WORD for j, t in enumerate(tokens)], tokens[position]


def mapping_function(example, lang):
    tokens = example['tokens']
    input_tokens, out = get_input_output(tokens, lang)
    return {'tokens': input_tokens, 'func_name': out, 'snippet': ' '.join(input_tokens)}


def filter_func_name(example):
    label = [l for l in example['func_name'].split('_') if l != '']
    return len(label) > 0


def main(args):
    input_dataset = load_dataset(args.input_dataset_hf)
    output_dataset = input_dataset.map(lambda example: mapping_function(example, args.lang))
    output_dataset = output_dataset.filter(filter_func_name)
    output_dataset.push_to_hub(args.output_dataset_hf, private=True)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset_hf', type=str, default='antolin/python-150_interduplication')
    parser.add_argument('--output_dataset_hf', type=str, default='antolin/python-150_func_interduplication')
    parser.add_argument('--lang', type=str, default='python')
    args = parser.parse_args()
    main(args)
