import pickle

import numpy as np
from matplotlib import pyplot as plt

DATA_CONTROL = 'bert_python-150.bin.pkl'
biases = []
tellies = list(range(0, 13))
for t in tellies:
    if t == 0:
        data_path = 'codebert_python-150.bin.pkl'
    else:
        data_path = f'codebert_python-150_telly{t}.bin.pkl'
    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)
    with open(DATA_CONTROL, 'rb') as handle:
        data_control = pickle.load(handle)

    diff_dup = [r - r_control for r, r_control in zip(data[1], data_control[1])]
    diff_no_dup = [r - r_control for r, r_control in zip(data[0], data_control[0])]
    bias = np.mean(diff_dup) - np.mean(diff_no_dup)
    biases.append(bias * 100)
    print(bias * 100, t)

plt.scatter(tellies, biases)
plt.ylim(-1, 3)
plt.show()

plt.savefig('telly_graph.png')

