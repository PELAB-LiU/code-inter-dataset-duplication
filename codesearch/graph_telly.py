import os.path
import pickle

import numpy as np
import pandas as pd
from plotnine import *
from scipy.stats import pearsonr, spearmanr

DATA_CONTROL = 'robertajava_python-150.bin.pkl'
biases = []
tellies = list(range(0, 13))
tellies_scatter = []
models = []
for t in tellies:
    for model in ['codebert', 'unixcoder', 'graphcodebert']:
        if t == 0:
            data_path = f'{model}_python-150.bin.pkl'
        else:
            data_path = f'{model}_python-150_telly{t}.bin.pkl'
        if not os.path.exists(data_path):
            continue
        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)
        with open(DATA_CONTROL, 'rb') as handle:
            data_control = pickle.load(handle)

        diff_dup = [r - r_control for r, r_control in zip(data[1], data_control[1])]
        diff_no_dup = [r - r_control for r, r_control in zip(data[0], data_control[0])]
        bias = np.mean(diff_dup) - np.mean(diff_no_dup)
        biases.append(bias * 100)
        tellies_scatter.append(t)
        models.append(model)
        print(t, bias * 100, np.mean(data[1] + data[0]), np.mean(data_control[1] + data_control[0]), model)


df = pd.DataFrame({'Layer': tellies_scatter, 'Bias': biases, 'Model': models})

# Calculate the correlation between the two lists
correlation = df['Layer'].corr(df['Bias'])

# Create a scatter plot with the correlation in the title
correlation_plot = (
    ggplot(df, aes(x='Layer', y='Bias', color='Model')) +
    geom_point() + stat_smooth(method='lm', se=False) +
    labs(title=f'Correlation: {correlation:.2f}')
)


print(f'Pearson: {pearsonr(tellies_scatter, biases)}')
print(f'Spearmanr: {spearmanr(tellies_scatter, biases)}')

correlation_plot.save(filename='correlation_plot.png')

