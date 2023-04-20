import argparse
import json

DATA = 'data.jsonl'
LANG = 'java'


def load_dataset(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    result = [json.loads(json_str) for json_str in json_list]
    return result


def main(args):
    dataset = load_dataset(DATA)
    print(len(dataset))
    print(dataset[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='../interduplication.db')
    args = parser.parse_args()
    main(args)
