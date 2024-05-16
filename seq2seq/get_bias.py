import argparse
import json

import numpy as np
import pandas as pd


def main(args):
    with open(args.judgement, 'r') as file:
        judgement = json.load(file)
    model = pd.read_csv(args.csv)

    no_dup = []
    dup = []
    for i in judgement:
        p, d = model[model['id'] == i]['performance'].iloc[0], model[model['id'] == i]['is_dup'].iloc[0]
        if d == 0:
            no_dup.append(p)
        else:
            dup.append(p)

    no_dup = np.mean(no_dup) * 100
    dup = np.mean(dup) * 100

    print(f'Dup: {dup:.2f} vs No dup: {no_dup:.2f}')
    print(f'Bias: {dup - no_dup:.2f}')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--judgement', type=str, required=True)
    args = parser.parse_args()
    main(args)
