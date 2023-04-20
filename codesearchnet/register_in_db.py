import argparse

from datasets import load_dataset


def main(args):
    dataset = load_dataset('code_search_net')
    print(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default='../interduplication.db')
    args = parser.parse_args()
    main(args)
